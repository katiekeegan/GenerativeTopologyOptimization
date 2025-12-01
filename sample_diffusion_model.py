#!/usr/bin/env python3
"""Sample a mesh by running reverse DDPM sampling in VAE latent space.

This script mirrors `sample_sdf_obj.py`'s output but obtains the latent by
running the trained diffusion model in latent space (z_T -> z_0), then
decodes z_0 with the VAE decoder into the conditioning vector used by the
SDF network, evaluates the SDF on a dense grid, runs marching cubes, and
exports a .obj file.

The script is defensive in loading checkpoints and supports a few common
model APIs (same style as `sample_sdf_obj.py`).
"""
import os
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from skimage import measure

from models import ImprovedVAE, ImprovedSDFNetwork, ModulationModule


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def sinusoidal_timestep_embedding(timesteps, dim: int):
    device = timesteps.device
    half = dim // 2
    log_max = math.log(10000.0)
    inv_freq = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -(log_max / (half - 1)))
    t = timesteps.float().unsqueeze(1).to(device)
    emb = t * inv_freq.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class MLPDiffusionModel(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 512, time_emb_dim: int = 128):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t_emb = sinusoidal_timestep_embedding(t, self.time_emb_dim)
        t_emb = t_emb.to(x.device)
        t_emb = self.time_mlp(t_emb)
        h = torch.cat([x, t_emb], dim=1)
        return self.net(h)


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


def reverse_diffusion_sample(diffusion_model, betas, alphas_cumprod, shape, device, timesteps=1000):
    diffusion_model.eval()
    betas = betas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)
    alphas = 1.0 - betas

    B = shape[0]
    z_t = torch.randn(shape, device=device)

    for t in reversed(range(timesteps)):
        t_batch = torch.full((B,), t, dtype=torch.long, device=device)
        with torch.no_grad():
            eps_pred = diffusion_model(z_t, t_batch)

        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        one_minus_alpha_cumprod_t = 1.0 - alpha_cumprod_t

        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (betas[t] / torch.sqrt(one_minus_alpha_cumprod_t))

        mean = coef1 * (z_t - coef2 * eps_pred)

        if t > 0:
            # posterior variance per DDPM
            alpha_cumprod_prev = alphas_cumprod[t - 1]
            var_t = betas[t] * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)
            noise = torch.randn_like(z_t)
            z_t = mean + torch.sqrt(var_t) * noise
        else:
            z_t = mean

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

    diffusion = MLPDiffusionModel(latent_dim=args.latent_dim, hidden=args.hidden_dim, time_emb_dim=args.time_emb_dim).to(device)
    load_diffusion_checkpoint(diffusion, args.diffusion_ckpt, device)

    # prepare diffusion schedule
    betas = cosine_beta_schedule(args.timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    # sample latent by reverse diffusion
    z_shape = (1, args.latent_dim)
    print(f"Sampling latent of shape {z_shape} with {args.timesteps} timesteps...")
    z0 = reverse_diffusion_sample(diffusion, betas, alphas_cumprod, z_shape, device, timesteps=args.timesteps)
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
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--encoding-dim', type=int, default=256)
    parser.add_argument('--chunk-size', type=int, default=200000)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--time-emb-dim', type=int, default=128)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--repair', action='store_true')
    parser.add_argument('--pad-boundary', action='store_true', help='conservatively pad boundary voxels from adjacent interior slices to enclose the surface')
    parser.add_argument('--boundary-pad-value', type=float, default=1.0, help='minimum magnitude used when padding boundary voxels')
    parser.add_argument('--no-cuda', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    sample_and_export(args)
