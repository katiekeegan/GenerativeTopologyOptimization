#!/usr/bin/env python3
"""Sample a mesh by running VE score-model sampling in VAE latent space.

This script mirrors `sample_sdf_obj.py`'s output but obtains the latent by
running the trained VE score model in latent space (z_T -> z_0), then decodes
z_0 with the VAE decoder into the conditioning vector used by the SDF network,
evaluates the SDF on a dense grid, runs marching cubes, and exports a .obj file.

The script is defensive in loading checkpoints and supports a few common
model APIs (same style as `sample_sdf_obj.py`).
"""
import os
import argparse
import math
import numpy as np
import torch
import trimesh
from skimage import measure

from models import ImprovedVAE, ImprovedSDFNetwork, ModulationModule
from trainer_diffusion import MLPDiffusionModel


def default_sigma_schedule(t: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    return sigma_min * torch.exp(math.log(sigma_max / sigma_min) * t)


def _call_sdf_network(sdf_network, query_points_chunk, z):
    try:
        out = sdf_network(query_points_chunk, z)
        return out
    except Exception as e1:
        try:
            B, C, _ = query_points_chunk.shape
            z_exp = z.unsqueeze(1).expand(-1, C, -1)
            out = sdf_network(query_points_chunk, z_exp)
            return out
        except Exception as e2:
            try:
                out = sdf_network(z, query_points_chunk)
                return out
            except Exception as e3:
                raise RuntimeError(
                    "sdf_network call failed in all tried orderings. Exceptions:\n"
                    f"1) sdf_network(query_points, z): {e1}\n"
                    f"2) sdf_network(query_points, z_expanded): {e2}\n"
                    f"3) sdf_network(z, query_points): {e3}\n"
                )


def evaluate_sdf_in_chunks(sdf_network, query_points, z, chunk_size=200_000):
    assert query_points.dim() == 3 and query_points.size(0) == 1, "query_points must be [1, N, 3]"
    device = query_points.device
    N = query_points.size(1)
    out_list = []
    with torch.no_grad():
        for i in range(0, N, chunk_size):
            j = min(N, i + chunk_size)
            q_chunk = query_points[:, i:j, :].to(device)
            out_chunk = _call_sdf_network(sdf_network, q_chunk, z)
            if out_chunk.dim() == 3 and out_chunk.size(-1) == 1:
                out_chunk = out_chunk.squeeze(-1)
            if out_chunk.dim() == 2:
                out_chunk = out_chunk[0]
            elif out_chunk.dim() == 1:
                out_chunk = out_chunk
            else:
                out_chunk = out_chunk.reshape(-1)
            out_list.append(out_chunk.detach().cpu())
    sdf_flat = torch.cat(out_list, dim=0).numpy().astype(np.float32)
    return sdf_flat


def reverse_ve_probability_flow_sample(
    score_model,
    shape,
    device,
    steps=1000,
    sigma_min=0.01,
    sigma_max=1.0,
    cond=None,
):
    """Sample z0 by integrating the VE probability-flow ODE backward.

    `trainer_diffusion.py` trains the model to predict the VE score
    s_theta(z_t, log sigma_t), so this sampler uses the same log-sigma time
    embedding instead of a DDPM epsilon-prediction reverse chain.
    """
    score_model.eval()
    B = shape[0]
    z_t = torch.randn(shape, device=device) * float(sigma_max)
    dt = 1.0 / float(steps)
    log_sigma_ratio = math.log(float(sigma_max) / float(sigma_min))

    for step in reversed(range(1, steps + 1)):
        t = torch.full((B,), step / float(steps), device=device)
        sigma_t = default_sigma_schedule(t, sigma_min, sigma_max)
        t_embed = torch.log(sigma_t)
        g2_t = 2.0 * log_sigma_ratio * sigma_t.pow(2)
        with torch.no_grad():
            score = score_model(z_t, t_embed, cond=cond)
        z_t = z_t + 0.5 * g2_t.unsqueeze(1) * score * dt

    return z_t


def load_modulation_checkpoint(modulation_module, ckpt_path, device):
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        try:
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                modulation_module.load_state_dict(ckpt['model_state_dict'])
            else:
                modulation_module.load_state_dict(ckpt)
            print(f"Loaded modulation module state_dict from {ckpt_path}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint into modulation_module: {e}. Proceeding with current init.")
    else:
        print(f"No modulation checkpoint found at {ckpt_path}; proceeding with random init.")


def load_diffusion_checkpoint(diffusion_model, ckpt_path, device):
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        try:
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                diffusion_model.load_state_dict(ckpt['model_state_dict'])
            elif isinstance(ckpt, dict) and 'ema_state_dict' in ckpt:
                diffusion_model.load_state_dict(ckpt['ema_state_dict'])
            else:
                diffusion_model.load_state_dict(ckpt)
            print(f"Loaded diffusion model from {ckpt_path}")
        except Exception as e:
            print(f"Warning: failed to load diffusion checkpoint: {e}. Proceeding with init.")
    else:
        print(f"No diffusion checkpoint at {ckpt_path}; proceeding with random init.")


def sample_and_export(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    # build models
    vae = ImprovedVAE(input_dim=args.encoding_dim, latent_dim=args.latent_dim, hidden_dim=512, num_layers=8).to(device)
    sdf_network = ImprovedSDFNetwork(input_dim=args.encoding_dim, latent_dim=args.encoding_dim, hidden_dim=128, output_dim=1, num_layers=8).to(device)
    modulation_module = ModulationModule(vae, sdf_network).to(device)

    load_modulation_checkpoint(modulation_module, args.modulation_ckpt, device)

    cond_dim = int(args.cond_dim) if int(args.cond_dim) > 0 else None
    diffusion = MLPDiffusionModel(latent_dim=args.latent_dim, hidden=args.hidden_dim, time_emb_dim=args.time_emb_dim, cond_dim=cond_dim).to(device)
    load_diffusion_checkpoint(diffusion, args.diffusion_ckpt, device)

    # sample latent by reverse VE probability-flow integration
    z_shape = (1, args.latent_dim)
    print(f"Sampling latent of shape {z_shape} with {args.timesteps} VE probability-flow steps...")
    z0 = reverse_ve_probability_flow_sample(
        diffusion,
        z_shape,
        device,
        steps=args.timesteps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )
    print("Sampling complete. Decoding with VAE decoder...")

    # decode to conditioning vector
    try:
        x_cond = modulation_module.vae.decoder(z0)
    except Exception:
        try:
            # maybe vae(z) returns (x_recon, z, ...)
            out = modulation_module.vae(z0)
            if isinstance(out, (tuple, list)) and len(out) >= 1:
                x_cond = out[0]
            else:
                x_cond = out
        except Exception:
            print("Failed to decode z with VAE decoder; using z directly as conditioning vector.")
            x_cond = z0

    # Build query grid
    lin = torch.linspace(-1.0, 1.0, args.grid, device=device)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing='ij'), dim=-1)
    query_points = grid.reshape(-1, 3).unsqueeze(0)

    sdf_flat = evaluate_sdf_in_chunks(modulation_module.sdf_network, query_points, x_cond, chunk_size=args.chunk_size)
    N = query_points.size(1)
    if sdf_flat.size != N:
        raise RuntimeError(f"Expected {N} SDF values but got {sdf_flat.size}")

    sdf_grid = sdf_flat.reshape(args.grid, args.grid, args.grid)
    print(f"SDF pred stats: min={sdf_grid.min():.6f}, max={sdf_grid.max():.6f}, mean={sdf_grid.mean():.6f}")
    # report boundary statistics to diagnose whether the surface touches the domain
    try:
        boundary_vals = np.concatenate([
            sdf_grid[0, :, :].ravel(), sdf_grid[-1, :, :].ravel(),
            sdf_grid[:, 0, :].ravel(), sdf_grid[:, -1, :].ravel(),
            sdf_grid[:, :, 0].ravel(), sdf_grid[:, :, -1].ravel()
        ])
        print(f"boundary SDF stats: min={boundary_vals.min():.6f}, max={boundary_vals.max():.6f}, mean={boundary_vals.mean():.6f}, std={boundary_vals.std():.6f}")
    except Exception as _e:
        print(f"warning computing boundary stats: {_e}")

    # optional conservative padding from interior slices (to enclose surface)
    if args.pad_boundary:
        def _pad_from_interior(grid, pad_value=1.0):
            # X faces
            interior = grid[1, :, :]
            mag = max(pad_value, np.abs(interior).max())
            grid[0, :, :] = np.sign(interior) * mag
            interior = grid[-2, :, :]
            mag = max(pad_value, np.abs(interior).max())
            grid[-1, :, :] = np.sign(interior) * mag
            # Y faces
            interior = grid[:, 1, :]
            mag = max(pad_value, np.abs(interior).max())
            grid[:, 0, :] = np.sign(interior) * mag
            interior = grid[:, -2, :]
            mag = max(pad_value, np.abs(interior).max())
            grid[:, -1, :] = np.sign(interior) * mag
            # Z faces
            interior = grid[:, :, 1]
            mag = max(pad_value, np.abs(interior).max())
            grid[:, :, 0] = np.sign(interior) * mag
            interior = grid[:, :, -2]
            mag = max(pad_value, np.abs(interior).max())
            grid[:, :, -1] = np.sign(interior) * mag
            return grid

        try:
            sdf_grid = _pad_from_interior(sdf_grid, pad_value=float(args.boundary_pad_value))
            print("Applied conservative boundary padding from interior slices.")
        except Exception as e:
            print(f"warning while padding boundary: {e}")

    if not (sdf_grid.min() < 0.0 < sdf_grid.max()):
        raise ValueError("SDF predictions do not cross zero — cannot extract surface.")

    # marching cubes: map indices [0..R-1] -> [-1,1] via spacing = 2 / (R-1)
    spacing = 2.0 / float(max(1, (args.grid - 1)))
    verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=0.0, spacing=(spacing, spacing, spacing))
    verts_world = verts + np.array([-1.0, -1.0, -1.0])

    # build mesh
    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
    mesh.fix_normals()

    # optional scaling
    if args.scale != 1.0:
        mesh.apply_scale(args.scale)

    # report watertightness and optionally attempt repair with diagnostics
    def mesh_diagnostics(m):
        try:
            v_count = int(m.vertices.shape[0])
            f_count = int(m.faces.shape[0])
            is_w = bool(m.is_watertight)
            euler = None
            try:
                euler = float(m.euler_number)
            except Exception:
                euler = None
            # boundary edges: edges appearing in only one face
            try:
                boundary_edge_count = int((m.edges_unique_face_count == 1).sum())
            except Exception:
                boundary_edge_count = None
            comp_count = None
            try:
                comp_count = len(m.split(only_watertight=False))
            except Exception:
                comp_count = None
            return {
                'verts': v_count,
                'faces': f_count,
                'is_watertight': is_w,
                'euler': euler,
                'boundary_edges': boundary_edge_count,
                'components': comp_count,
            }
        except Exception as _e:
            return {'error': str(_e)}

    diag_before = mesh_diagnostics(mesh)
    print(f"mesh diagnostics before repair: {diag_before}")

    if args.repair:
        try:
            # try several conservative cleanup steps
            # 1) remove duplicate faces
            try:
                mesh.remove_duplicate_faces()
            except Exception:
                pass
            # 2) remove unreferenced vertices
            try:
                mesh.remove_unreferenced_vertices()
            except Exception:
                pass
            # 3) remove degenerate faces via update_faces(nondegenerate_faces())
            try:
                mesh.update_faces(mesh.nondegenerate_faces())
            except Exception:
                pass
            # 4) merge very close vertices
            try:
                mesh.merge_vertices()
            except Exception:
                pass
            # 5) fill holes using trimesh.repair.fill_holes
            try:
                import trimesh.repair as trepair
                trepair.fill_holes(mesh)
            except Exception:
                pass

            # re-fix normals
            try:
                mesh.fix_normals()
            except Exception:
                pass

            # recompute diagnostics
            diag_after = mesh_diagnostics(mesh)
            print(f"mesh diagnostics after trimesh repair steps: {diag_after}")

            # If still not watertight, try pymeshfix if available (stronger repair)
            if not mesh.is_watertight:
                try:
                    from pymeshfix import MeshFix
                    mf = MeshFix(mesh.vertices.copy(), mesh.faces.copy())
                    mf.repair()
                    mesh = trimesh.Trimesh(vertices=mf.v.copy(), faces=mf.f.copy(), process=False)
                    try:
                        mesh.fix_normals()
                    except Exception:
                        pass
                    diag_pf = mesh_diagnostics(mesh)
                    print(f"mesh diagnostics after pymeshfix repair: {diag_pf}")
                except Exception as e:
                    print(f"pymeshfix not available or failed: {e}")

            print(f"mesh watertight after repair attempts: {mesh.is_watertight}")
        except Exception as e:
            print(f"mesh repair warning: {e}")

    mesh.export(args.outfile)
    print(f"Saved sampled mesh to: {args.outfile}")


def parse_args():
    parser = argparse.ArgumentParser(description='Sample SDF mesh from diffusion model in latent space')
    parser.add_argument('--modulation-ckpt', type=str, default='checkpoints_mod/mod_last.pth')
    parser.add_argument('--diffusion-ckpt', type=str, default='checkpoints_diffusion/diffusion_epoch_1000.pth')
    parser.add_argument('--grid', type=int, default=64)
    parser.add_argument('--outfile', type=str, default='sampled_diffusion_shape.obj')
    parser.add_argument('--timesteps', type=int, default=1000, help='number of VE probability-flow reverse steps')
    parser.add_argument('--sigma-min', type=float, default=0.01, help='VE sigma_min used during score training')
    parser.add_argument('--sigma-max', type=float, default=1.0, help='VE sigma_max used during score training')
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--encoding-dim', type=int, default=256)
    parser.add_argument('--chunk-size', type=int, default=200000)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--time-emb-dim', type=int, default=128)
    parser.add_argument('--cond-dim', type=int, default=0, help='set to 8 for checkpoints trained with --cond; uses zero conditioning at sample time')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--repair', action='store_true')
    parser.add_argument('--pad-boundary', action='store_true', help='conservatively pad boundary voxels from adjacent interior slices to enclose the surface')
    parser.add_argument('--boundary-pad-value', type=float, default=1.0, help='minimum magnitude used when padding boundary voxels')
    parser.add_argument('--no-cuda', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    sample_and_export(args)
