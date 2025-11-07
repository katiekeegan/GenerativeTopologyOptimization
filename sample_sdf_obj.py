#!/usr/bin/env python3
"""
Sample an SDF from the VAE prior and export a mesh (.obj).

This script:
- samples z ~ N(0, prior_sigma^2) with shape [1, latent_dim]
- builds a dense query grid in [-1,1]^3 of size R^3
- evaluates the SDF network in chunks (safe for memory)
- runs marching cubes and maps verts back to [-1,1]^3
- exports the mesh as an OBJ

This file is intentionally defensive and trainer-compatible:
- it calls sdf_network(query_points, z) (query_points first), which is how trainer.py uses the SDF network.
- if the SDF network expects per-point latent vectors, the script will tile z to match the chunk length automatically.
"""
import os
import argparse
import numpy as np
import torch
import trimesh
from skimage import measure

from models import ImprovedVAE, ImprovedSDFNetwork, ModulationModule

def _call_sdf_network(sdf_network, query_points_chunk, z):
    """
    Try to call sdf_network in a robust way.
    Preferred order (trainer.py): sdf_network(query_points, z)
    Fallback 1: if network expects per-point z, tile z to [B, C, latent_dim] and call sdf_network(query_points, z_exp)
    Fallback 2: try sdf_network(z, query_points) (other implementers)
    Returns tensor with shape [B, C] or [B, C, 1]
    """
    # Try preferred ordering first
    try:
        out = sdf_network(query_points_chunk, z)
        return out
    except Exception as e1:
        # Try tiling z per-point (some networks expect z of shape [B, C, latent_dim])
        try:
            B, C, _ = query_points_chunk.shape
            z_exp = z.unsqueeze(1).expand(-1, C, -1)  # [B, C, latent_dim]
            out = sdf_network(query_points_chunk, z_exp)
            return out
        except Exception as e2:
            # Fallback: try swapping arguments
            try:
                out = sdf_network(z, query_points_chunk)
                return out
            except Exception as e3:
                # Raise a combined error for debugging
                raise RuntimeError(
                    "sdf_network call failed in all tried orderings. Exceptions:\n"
                    f"1) sdf_network(query_points, z): {e1}\n"
                    f"2) sdf_network(query_points, z_expanded): {e2}\n"
                    f"3) sdf_network(z, query_points): {e3}\n"
                )

def evaluate_sdf_in_chunks(sdf_network, query_points, z, chunk_size=200_000):
    """
    Evaluate the sdf_network on query_points in chunks.

    Args:
        sdf_network: callable accepting (query_points_chunk, z) or variants.
        query_points: tensor [1, N, 3] (N = R^3)
        z: tensor [1, latent_dim] (or [1, N, latent_dim] in some impls)
        chunk_size: number of query points to evaluate at once
    Returns:
        sdf_flat: 1D numpy array length N with SDF values
    """
    assert query_points.dim() == 3 and query_points.size(0) == 1, "query_points must be [1, N, 3]"
    device = query_points.device
    N = query_points.size(1)
    out_list = []
    with torch.no_grad():
        for i in range(0, N, chunk_size):
            j = min(N, i + chunk_size)
            q_chunk = query_points[:, i:j, :].to(device)  # [1, C, 3]
            # Call robust wrapper
            out_chunk = _call_sdf_network(sdf_network, q_chunk, z)
            # normalize output shapes: accept [B,C,1], [B,C] etc.
            if out_chunk.dim() == 3 and out_chunk.size(-1) == 1:
                out_chunk = out_chunk.squeeze(-1)
            if out_chunk.dim() == 2:
                # [B,C] -> take first batch
                out_chunk = out_chunk[0]
            elif out_chunk.dim() == 1:
                # single-dim vector (C,) - acceptable
                out_chunk = out_chunk
            else:
                # try to flatten
                out_chunk = out_chunk.reshape(-1)
            out_list.append(out_chunk.detach().cpu())
    sdf_flat = torch.cat(out_list, dim=0).numpy().astype(np.float32)
    return sdf_flat

def sample_sdf_from_prior_and_save(modulation_module, device, filename="sampled_shape.obj",
                                   grid_resolution=128, prior_sigma=0.25, latent_dim=None,
                                   chunk_size=200_000, out_scale=1.0):
    """
    Main routine to sample SDF and save mesh.
    """
    modulation_module.eval()

    # Build query grid in [-1,1]^3
    lin = torch.linspace(-1.0, 1.0, grid_resolution, device=device)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), dim=-1)  # [R,R,R,3]
    query_points = grid.reshape(-1, 3).unsqueeze(0)  # [1, R^3, 3]
    N = query_points.size(1)
    print(f"[sample] Query grid resolution {grid_resolution} -> {N} points")

    # latent dim
    if latent_dim is None:
        # try modulation_module.vae.latent_dim
        if hasattr(modulation_module, "vae") and hasattr(modulation_module.vae, "latent_dim"):
            latent_dim = int(modulation_module.vae.latent_dim)
        else:
            raise RuntimeError("latent_dim must be provided or modulation_module.vae.latent_dim must exist")

    # sample z
    z = torch.randn(1, latent_dim, device=device) * float(prior_sigma)  # [1, latent_dim]
    print(f"[sample] sampled z shape {z.shape}, prior sigma {prior_sigma}")

    # Evaluate SDF in chunks using the robust evaluator (calls in trainer style)
    sdf_flat = evaluate_sdf_in_chunks(modulation_module.sdf_network, query_points, z, chunk_size=chunk_size)
    if sdf_flat.size != N:
        raise RuntimeError(f"Expected {N} SDF values but got {sdf_flat.size}")

    # reshape to grid
    sdf_grid = sdf_flat.reshape(grid_resolution, grid_resolution, grid_resolution)
    print(f"[sample] SDF pred stats: min={sdf_grid.min():.6f}, max={sdf_grid.max():.6f}, mean={sdf_grid.mean():.6f}")

    if not (sdf_grid.min() < 0.0 < sdf_grid.max()):
        raise ValueError("SDF predictions do not cross zero — cannot extract surface.")

    # marching cubes: map indices [0..R-1] -> [-1,1] via spacing = 2 / (R-1)
    spacing = 2.0 / float(max(1, (grid_resolution - 1)))
    verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=0.0, spacing=(spacing, spacing, spacing))
    # marching_cubes returns coords = index * spacing. Map to [-1,1] by adding origin = -1
    verts_world = verts + np.array([-1.0, -1.0, -1.0])

    # build mesh
    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
    mesh.rezero()
    mesh.fix_normals()

    # optional scaling
    if out_scale is not None and float(out_scale) != 1.0:
        mesh.apply_scale(float(out_scale))

    mesh.export(filename)
    print(f"[sample] Saved sampled mesh to: {filename}")
    return filename

def main():
    parser = argparse.ArgumentParser(description="Sample SDF from VAE prior and export a mesh (.obj)")
    parser.add_argument("--encoding-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=4096, help="VAE input / flattened latent dimension")
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--outfile", type=str, default="sampled_shape.obj")
    parser.add_argument("--prior-sigma", type=float, default=0.25)
    parser.add_argument("--ckpt", type=str, default="checkpoints_mod/mod_last.pth")
    parser.add_argument("--chunk-size", type=int, default=200000, help="chunk size for SDF evaluation")
    parser.add_argument("--scale", type=float, default=1.0, help="optional output mesh scale")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build models (same construction as trainer.py)
    vae = ImprovedVAE(input_dim=args.latent_dim, latent_dim=args.encoding_dim, hidden_dim=1024, num_layers=8).to(device)
    sdf_network = ImprovedSDFNetwork(input_dim=args.encoding_dim, latent_dim=args.latent_dim, hidden_dim=512, output_dim=1, num_layers=8).to(device)
    modulation_module = ModulationModule(vae, sdf_network).to(device)

    # load checkpoint if present; robust to two checkpoint formats
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        try:
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                modulation_module.load_state_dict(ckpt["model_state_dict"])
            else:
                modulation_module.load_state_dict(ckpt)
            print(f"Loaded modulation module state_dict from {args.ckpt}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint into modulation_module: {e}. Proceeding with current init.")
    else:
        print(f"No checkpoint found at {args.ckpt}; proceeding with randomly initialized model (not recommended).")

    sample_sdf_from_prior_and_save(modulation_module, device,
                                   filename=args.outfile,
                                   grid_resolution=int(args.grid),
                                   prior_sigma=float(args.prior_sigma),
                                   latent_dim=int(args.latent_dim),
                                   chunk_size=int(args.chunk_size),
                                   out_scale=float(args.scale))

if __name__ == "__main__":
    main()