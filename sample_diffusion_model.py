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
from dl4to.datasets import SELTODataset
from utils.preprocess_data import (
    create_problem_information_lists,
    create_voxel_grids,
    VoxelSDFDataset,
    collate_fn,
)
from torch.utils.data import DataLoader


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


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(self.norm1(h))
        h = self.drop(h)
        h = self.fc2(h)
        h = self.norm2(h)
        return self.act(x + h)


class MLPDiffusionModel(nn.Module):
    """Match the architecture used in trainer_diffusion.py for checkpoint compatibility.
    Stronger score net with residual blocks, LayerNorm, SiLU; optional conditioning.
    """
    def __init__(self, latent_dim: int, hidden: int = 512, time_emb_dim: int = 128, cond_dim: int | None = None, num_blocks: int = 6, dropout: float = 0.0):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim

        # Time embedding MLP: [time_emb_dim -> hidden -> time_emb_dim]
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, time_emb_dim),
        )

        in_dim = latent_dim + time_emb_dim + (cond_dim or 0)
        self.input = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden, dropout=dropout) for _ in range(num_blocks)])
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

        # For backward compatibility when caller passes integer timesteps, we will embed internally

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, cond: torch.Tensor | None = None):
        # t_embed: real-valued scalar per batch (e.g., log-sigma or normalized time)
        t_emb = sinusoidal_timestep_embedding(t_embed, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb.to(x.device))

        # Only use conditioning if the model was instantiated with cond_dim > 0
        if cond is not None and (self.cond_dim is not None and self.cond_dim > 0):
            if cond.dim() == 5:
                cond_vec = cond.mean(dim=(-1, -2, -3))
            elif cond.dim() == 2:
                cond_vec = cond
            else:
                cond_vec = cond.view(cond.size(0), -1)
            cond_vec = cond_vec.to(x.device)
            h_in = torch.cat([x, t_emb, cond_vec], dim=1)
        else:
            if self.cond_dim is not None and self.cond_dim > 0:
                cond_vec = torch.zeros(x.size(0), self.cond_dim, device=x.device, dtype=x.dtype)
                h_in = torch.cat([x, t_emb, cond_vec], dim=1)
            else:
                h_in = torch.cat([x, t_emb], dim=1)

        h = self.input(h_in)
        for blk in self.blocks:
            h = blk(h)
        return self.output(h)


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


def reverse_diffusion_sample(diffusion_model, betas, alphas_cumprod, shape, device, timesteps=1000, cond_vec: torch.Tensor | None = None):
    diffusion_model.eval()
    betas = betas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)
    alphas = 1.0 - betas

    B = shape[0]
    z_t = torch.randn(shape, device=device)

    for t in reversed(range(timesteps)):
        t_batch = torch.full((B,), t / float(max(1, timesteps)), dtype=torch.float32, device=device)
        with torch.no_grad():
            eps_pred = diffusion_model(z_t, t_batch, cond=cond_vec)

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

    # If cond_dim is not provided (>0), we will infer it from SELTO and get a default cond
    inferred_cond_dim = args.cond_dim
    cond_vec = None

    if args.use_selto_cond:
        # Load SELTO and build voxel dataset similar to trainer_posterior.py
        selto = SELTODataset(root=".", name=args.dataset_name, train=True)
        voxel_grids = create_voxel_grids(selto)
        F_list, Omega_list = create_problem_information_lists(selto)
        problem_information_list = (F_list, Omega_list)
        dataset = VoxelSDFDataset(
            voxel_grids,
            problem_information_list=problem_information_list,
            num_query_points=args.num_query_points,
            fixed_surface_points_size=args.fixed_surface_points_size,
            noise_std=0.0,
            device=device,
            dataset="SELTO",
            return_problem_information=True,
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        first_batch = next(iter(dataloader))
        # Expect (pc, qp, sdf_vals, cond_grid, F, Ω_design)
        if len(first_batch) != 6:
            raise RuntimeError("Expected 6-tuple from collate_fn: pc, qp, sdf_vals, cond, F, Ω_design")
        _, _, _, cond_grid, _, _ = first_batch
        if isinstance(cond_grid, list):
            cond_grid = torch.stack(cond_grid, dim=0)
        # cond_grid shape [B=1, C_cond, D, H, W]
        inferred_cond_dim = int(cond_grid.shape[1])
        # Global average pool to a [1, C_cond] vector
        cond_first = cond_grid[0:1]
        cond_vec = cond_first.view(cond_first.size(1), -1).mean(dim=1, keepdim=True).t().to(device)

    # Determine cond_dim from checkpoint input layer if available to avoid size mismatches
    cond_dim_from_ckpt = None
    if os.path.exists(args.diffusion_ckpt):
        try:
            ckpt = torch.load(args.diffusion_ckpt, map_location=device)
            state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
            # Trainer uses 'input.0.weight' for first linear layer; its second dim is in_dim
            w = state.get('input.0.weight', None)
            if isinstance(w, torch.Tensor):
                in_dim_ckpt = w.shape[1]
                base_in = args.latent_dim + args.time_emb_dim
                cond_dim_from_ckpt = max(0, in_dim_ckpt - base_in)
        except Exception:
            cond_dim_from_ckpt = None

    effective_cond_dim = inferred_cond_dim
    if cond_dim_from_ckpt is not None:
        if cond_dim_from_ckpt != inferred_cond_dim:
            print(f"Info: checkpoint expects cond_dim={cond_dim_from_ckpt}, overriding inferred={inferred_cond_dim}.")
        effective_cond_dim = cond_dim_from_ckpt

    # If checkpoint expects no conditioning, drop any precomputed cond_vec
    if effective_cond_dim == 0:
        cond_vec = None

    diffusion = MLPDiffusionModel(latent_dim=args.latent_dim, hidden=args.hidden_dim, time_emb_dim=args.time_emb_dim, cond_dim=effective_cond_dim).to(device)
    if os.path.exists(args.diffusion_ckpt):
        load_diffusion_checkpoint(diffusion, args.diffusion_ckpt, device)
    else:
        print(f"No diffusion checkpoint at {args.diffusion_ckpt}; proceeding with random init.")

    # prepare conditioning vector (optional)
    if effective_cond_dim > 0 and cond_vec is None:
        if args.cond_file and os.path.exists(args.cond_file):
            try:
                cond_grid = np.load(args.cond_file)
                # Expect shape [C, D, H, W] or [1, C, D, H, W]; pool to [C]
                if cond_grid.ndim == 5:
                    cond_grid = cond_grid[0]
                if cond_grid.ndim != 4:
                    raise ValueError(f"cond file must be 4D [C,D,H,W], got shape {cond_grid.shape}")
                cond_channels = cond_grid.shape[0]
                if cond_channels != effective_cond_dim:
                    print(f"Warning: effective cond_dim ({effective_cond_dim}) != channels in cond_file ({cond_channels}); will pad/truncate.")
                pooled = cond_grid.reshape(cond_channels, -1).mean(axis=1).astype(np.float32)
                # pad or truncate to cond_dim
                if cond_channels < effective_cond_dim:
                    pooled = np.pad(pooled, (0, effective_cond_dim - cond_channels))
                else:
                    pooled = pooled[:effective_cond_dim]
                cond_vec = torch.from_numpy(pooled).unsqueeze(0).to(device)
            except Exception as e:
                print(f"Warning: failed to read cond_file '{args.cond_file}': {e}; using zeros.")
                cond_vec = torch.zeros(1, effective_cond_dim, device=device)
        else:
            # default: zero conditioning
            cond_vec = torch.zeros(1, effective_cond_dim, device=device)

    # prepare diffusion schedule
    betas = cosine_beta_schedule(args.timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    # sample latent by reverse diffusion
    z_shape = (1, args.latent_dim)
    print(f"Sampling latent of shape {z_shape} with {args.timesteps} timesteps...")
    z0 = reverse_diffusion_sample(diffusion, betas, alphas_cumprod, z_shape, device, timesteps=args.timesteps, cond_vec=cond_vec)
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
    parser.add_argument('--diffusion-ckpt', type=str, default='checkpoints_diffusion/diffusion_epoch_10000.pth')
    parser.add_argument('--grid', type=int, default=64)
    parser.add_argument('--outfile', type=str, default='sampled_diffusion_shape.obj')
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--encoding-dim', type=int, default=256)
    parser.add_argument('--chunk-size', type=int, default=200000)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--time-emb-dim', type=int, default=128)
    parser.add_argument('--cond-dim', type=int, default=8, help='Conditioning vector dimension; if --use-selto-cond, inferred from dataset')
    parser.add_argument('--cond-file', type=str, default='', help='Optional path to a .npy file containing a conditioning grid [C,D,H,W] to pool into a vector')
    parser.add_argument('--use-selto-cond', action='store_true', help='Load SELTO and use the first sample conditioning as default cond')
    parser.add_argument('--dataset-name', type=str, default='sphere_complex', help='SELTO dataset name (e.g., sphere_complex, disc_simple)')
    parser.add_argument('--num-query-points', type=int, default=5000)
    parser.add_argument('--fixed-surface-points-size', type=int, default=5000)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--repair', action='store_true')
    parser.add_argument('--pad-boundary', action='store_true', help='conservatively pad boundary voxels from adjacent interior slices to enclose the surface')
    parser.add_argument('--boundary-pad-value', type=float, default=1.0, help='minimum magnitude used when padding boundary voxels')
    parser.add_argument('--no-cuda', action='store_true')
    # Match training defaults: enable SELTO-based conditioning by default
    parser.set_defaults(use_selto_cond=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    sample_and_export(args)
