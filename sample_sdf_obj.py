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
from torch.utils.data import DataLoader
from dl4to.datasets import SELTODataset
from utils.preprocess_data import create_voxel_grids, VoxelSDFDataset, collate_fn
from utils.preprocess_data import repair_mesh

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
                                   chunk_size=200_000, out_scale=1.0,
                                   pad_boundary=False, boundary_pad_value=1.0, repair_mesh=False):
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

    # Decode sampled latent into the same conditioning vector used during training
    # (during training the modulation module conditions the SDF network on the
    # VAE decoder output `x_recon`, not the raw latent `z`). Use the decoder to
    # obtain the conditioning vector so sampling matches training-time behavior.
    try:
        x_cond = modulation_module.vae.decoder(z)
    except Exception:
        # Fallback: if the VAE API differs, try calling modulation_module.vae(z)
        try:
            _, x_cond, _, _, _ = modulation_module.vae(z)
        except Exception:
            x_cond = z

    # Evaluate SDF in chunks using the robust evaluator (calls in trainer style)
    sdf_flat = evaluate_sdf_in_chunks(modulation_module.sdf_network, query_points, x_cond, chunk_size=chunk_size)
    if sdf_flat.size != N:
        raise RuntimeError(f"Expected {N} SDF values but got {sdf_flat.size}")

    # reshape to grid
    sdf_grid = sdf_flat.reshape(grid_resolution, grid_resolution, grid_resolution)
    print(f"[sample] SDF pred stats: min={sdf_grid.min():.6f}, max={sdf_grid.max():.6f}, mean={sdf_grid.mean():.6f}")

    # report boundary statistics to diagnose whether the surface touches the domain
    try:
        boundary_vals = np.concatenate([
            sdf_grid[0, :, :].ravel(), sdf_grid[-1, :, :].ravel(),
            sdf_grid[:, 0, :].ravel(), sdf_grid[:, -1, :].ravel(),
            sdf_grid[:, :, 0].ravel(), sdf_grid[:, :, -1].ravel()
        ])
        print(f"[sample] boundary SDF stats: min={boundary_vals.min():.6f}, max={boundary_vals.max():.6f}, mean={boundary_vals.mean():.6f}, std={boundary_vals.std():.6f}")
    except Exception as _e:
        print(f"[sample] warning computing boundary stats: {_e}")

    # optional conservative padding: replace boundary voxels with sign-preserving values derived
    # from the immediately interior slice. This avoids sampling outside the trained domain while
    # ensuring the iso-surface is enclosed.
    if pad_boundary:
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
            sdf_grid = _pad_from_interior(sdf_grid, pad_value=float(boundary_pad_value))
            print("[sample] Applied conservative boundary padding from interior slices.")
        except Exception as e:
            print(f"[sample] warning while padding boundary: {e}")

    if not (sdf_grid.min() < 0.0 < sdf_grid.max()):
        raise ValueError("SDF predictions do not cross zero — cannot extract surface.")

    # marching cubes: map indices [0..R-1] -> [-1,1] via spacing = 2 / (R-1)
    spacing = 2.0 / float(max(1, (grid_resolution - 1)))
    verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=0.0, spacing=(spacing, spacing, spacing))
    # marching_cubes returns coords = index * spacing. Map to [-1,1] by adding origin = -1
    verts_world = verts + np.array([-1.0, -1.0, -1.0])

    # build mesh
    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
    # Do not recenter the mesh automatically here (mesh.rezero()). Keep original coords.
    mesh.fix_normals()

    # optional scaling
    if out_scale is not None and float(out_scale) != 1.0:
        mesh.apply_scale(float(out_scale))

    # report watertightness and optionally attempt repair
    try:
        print(f"[sample] mesh watertight before repair: {mesh.is_watertight}")
    except Exception:
        print("[sample] mesh watertight status unknown")

    if repair_mesh:
        try:
            import trimesh.repair as repair
            # attempt to fill holes and clean small defects
            repair.fill_holes(mesh)
            mesh.remove_degenerate_faces()
            mesh.merge_vertices()
            print(f"[sample] attempted mesh repair; watertight now: {mesh.is_watertight}")
        except Exception as e:
            print(f"[sample] mesh repair warning: {e}")

    mesh.export(filename)
    print(f"[sample] Saved sampled mesh to: {filename}")
    return filename

def compute_true_sdf_stats(root='.', name='sphere_complex', device='cpu', max_values=200000):
    """
    Compute simple statistics over ground-truth SDFs from the SELTO dataset via VoxelSDFDataset.

    Returns a dict with min/max/mean/std/median/percentiles and mean-abs.
    This streams through the dataset and samples up to `max_values` SDF scalars for percentile estimates.
    """
    selto = SELTODataset(root=root, name=name, train=True)
    voxel_grids = create_voxel_grids(selto)
    # create a VoxelSDFDataset that mirrors training usage but on CPU
    dataset = VoxelSDFDataset(voxel_grids, num_query_points=5000, fixed_surface_points_size=5000, noise_std=0.0, device=device)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    vals = []
    total = 0
    minv = float('inf')
    maxv = float('-inf')
    # streaming mean/std (Welford)
    mean = 0.0
    m2 = 0.0
    count = 0

    for _, _, sdf in loader:
        arr = sdf.view(-1).cpu().numpy()
        if arr.size == 0:
            continue
        total += arr.size
        # update min/max
        minv = min(minv, float(arr.min()))
        maxv = max(maxv, float(arr.max()))
        # update Welford
        for x in arr:
            count += 1
            delta = x - mean
            mean += delta / count
            delta2 = x - mean
            m2 += delta * delta2

        # collect sample for percentiles up to cap
        if len(vals) < max_values:
            remaining = max_values - len(vals)
            if arr.size <= remaining:
                vals.append(arr)
            else:
                vals.append(arr[:remaining])
        if len(vals) >= max_values:
            break

    if count < 1:
        raise RuntimeError("No SDF values found in dataset to compute statistics")

    std = (m2 / (count - 1)) ** 0.5 if count > 1 else 0.0
    all_sample = np.concatenate(vals) if len(vals) > 0 else np.array([mean])
    median = float(np.median(all_sample))
    p10 = float(np.percentile(all_sample, 10))
    p90 = float(np.percentile(all_sample, 90))
    mean_abs = float(np.mean(np.abs(all_sample)))

    stats = {
        'min': float(minv),
        'max': float(maxv),
        'mean_stream': float(mean),
        'std_stream': float(std),
        'median_sample': median,
        'p10_sample': p10,
        'p90_sample': p90,
        'mean_abs_sample': mean_abs,
        'count_sampled': int(len(all_sample)),
        'total_seen': int(total),
    }
    return stats


def save_dataset_sample_sdfs(modulation_module, device, root='.', name='sphere_complex',
                             sample_idx=0, num_query_points=5000, fixed_surface_points_size=5000,
                             out_prefix='sample', pad_boundary=False, boundary_pad_value=1.0,
                             mesh_grid_resolution=64, mesh_chunk_size=20000, repair_mesh_flag=False):
    """
    Pull a single sample from VoxelSDFDataset, evaluate the modulation pipeline on the
    dataset's query points, and save both the ground-truth SDF scalars and the generated SDF
    predictions to an .npz file for later analysis.
    """
    selto = SELTODataset(root=root, name=name, train=True)
    voxel_grids = create_voxel_grids(selto)
    dataset = VoxelSDFDataset(voxel_grids, num_query_points=int(num_query_points),
                              fixed_surface_points_size=int(fixed_surface_points_size),
                              noise_std=0.0, device=device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # fetch requested sample
    for i, batch in enumerate(loader):
        if i != int(sample_idx):
            continue
        # collate_fn returns (point_clouds:list, query_points:Tensor[B,N,3], sdf_values:Tensor[B,N])
        try:
            point_cloud_batch, query_points_batch, sdf_gt_batch = batch
        except Exception:
            if len(batch) >= 3:
                point_cloud_batch, query_points_batch, sdf_gt_batch = batch[0], batch[1], batch[2]
            else:
                raise RuntimeError("Unexpected batch format from VoxelSDFDataset collate_fn")
        break
    else:
        raise IndexError(f"Requested sample index {sample_idx} out of range")

    modulation_module.eval()
    # point_cloud_batch from collate_fn is a list of tensors (one per batch item).
    # Convert to a single tensor batch if needed.
    if isinstance(point_cloud_batch, list):
        try:
            point_cloud_batch = torch.stack(point_cloud_batch, dim=0)
        except Exception:
            # fallback: take first element
            point_cloud_batch = point_cloud_batch[0].unsqueeze(0)
    point_cloud_batch = point_cloud_batch.to(device)
    query_points_batch = query_points_batch.to(device)

    # get conditioning vector x_cond from VAE in a robust way
    x_cond = None
    try:
        vae_out = modulation_module.vae(point_cloud_batch)
        if isinstance(vae_out, tuple) and len(vae_out) >= 1:
            # ImprovedVAE returns (x_recon, z, latent_pc, mu, logvar)
            # x_recon is element 0 (conditioning vector); element 1 is z (latent).
            x_cond = vae_out[0]
        else:
            x_cond = vae_out
    except Exception as e:
        # Try calling encoder+decoder if available
        try:
            z = modulation_module.vae.encode(point_cloud_batch)
            x_cond = modulation_module.vae.decoder(z)
        except Exception:
            print(f"[sample_dataset] warning: failed to obtain conditioning vector from VAE: {e}")
            x_cond = None

    # If we couldn't get x_cond, try to call modulation_module directly (some wrappers expose this)
    sdf_pred_np = None
    if x_cond is None:
        try:
            # Try modulation_module API: modulation_module(point_cloud, query_points) -> sdf
            with torch.no_grad():
                sdf_pred = modulation_module(point_cloud_batch, query_points_batch)
            sdf_pred_np = sdf_pred.detach().cpu().numpy().ravel()
        except Exception as e:
            raise RuntimeError(f"Failed to get generated SDF via modulation_module: {e}")
    else:
        # Ensure x_cond has batch dimension and shape [1, latent_dim]
        if isinstance(x_cond, torch.Tensor):
            if x_cond.dim() == 1:
                x_cond = x_cond.unsqueeze(0)
            elif x_cond.dim() == 2 and x_cond.size(0) != 1:
                # If a batch is present, take first element
                x_cond = x_cond[:1]
        # Evaluate sdf_network on the dataset's query points using existing evaluator
        sdf_pred_np = evaluate_sdf_in_chunks(modulation_module.sdf_network, query_points_batch, x_cond)

    sdf_gt_np = sdf_gt_batch.detach().cpu().numpy().ravel()
    qpts_np = query_points_batch.detach().cpu().numpy().reshape(-1, 3)

    out_file = f"{out_prefix}_sample{int(sample_idx)}_sdfs.npz"
    np.savez_compressed(out_file, query_points=qpts_np, sdf_gt=sdf_gt_np, sdf_pred=sdf_pred_np)
    print(f"[sample_dataset] Saved GT & predicted SDFs to: {out_file}")
    # Print simple diagnostics
    print(f"[sample_dataset] gt min/max: {sdf_gt_np.min():.6f}/{sdf_gt_np.max():.6f}, pred min/max: {sdf_pred_np.min():.6f}/{sdf_pred_np.max():.6f}")
    # --- Export ground-truth OBJ from voxel grid ---
    try:
        voxel = voxel_grids[int(sample_idx)]
        try:
            voxel_np = voxel.cpu().numpy()
        except Exception:
            voxel_np = np.array(voxel)
        try:
            verts, faces, _, _ = measure.marching_cubes(voxel_np, level=0.5)
            verts_np = verts.copy()
            try:
                verts_np = dataset._permute_verts_if_needed(verts_np)
            except Exception:
                pass
            verts_t = torch.tensor(verts_np, dtype=torch.float32)
            verts_normalized = (verts_t - dataset.center) / dataset.coord_scale
            verts_world = verts_normalized.numpy()
            mesh_gt = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
            try:
                mesh_gt = repair_mesh(mesh_gt)
            except Exception:
                pass
            gt_obj = f"{out_prefix}_sample{int(sample_idx)}_gt.obj"
            mesh_gt.export(gt_obj)
            print(f"[sample_dataset] Saved ground-truth OBJ to: {gt_obj}")
        except Exception as e:
            print(f"[sample_dataset] warning: failed to build GT mesh from voxel grid: {e}")
    except Exception as e:
        print(f"[sample_dataset] warning accessing voxel grids for GT mesh: {e}")

    # --- Export predicted OBJ by evaluating dense grid conditioned on x_cond ---
    try:
        # Build dense query grid in [-1,1]^3
        lin = torch.linspace(-1.0, 1.0, int(mesh_grid_resolution), device=device)
        grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), dim=-1)  # [R,R,R,3]
        query_dense = grid.reshape(-1, 3).unsqueeze(0)  # [1, R^3, 3]
        N = query_dense.size(1)
        print(f"[sample_dataset] Evaluating predicted SDF on dense grid {mesh_grid_resolution} -> {N} points")
        sdf_flat_pred = evaluate_sdf_in_chunks(modulation_module.sdf_network, query_dense, x_cond, chunk_size=int(mesh_chunk_size))
        sdf_grid_pred = sdf_flat_pred.reshape(mesh_grid_resolution, mesh_grid_resolution, mesh_grid_resolution)
        # optional conservative padding
        if pad_boundary:
            try:
                sdf_grid_pred = _pad_from_interior(sdf_grid_pred, pad_value=float(boundary_pad_value))
            except Exception:
                pass
        if not (sdf_grid_pred.min() < 0.0 < sdf_grid_pred.max()):
            print("[sample_dataset] warning: predicted dense SDF does not cross zero; skipping predicted OBJ export")
        else:
            spacing = 2.0 / float(max(1, (mesh_grid_resolution - 1)))
            verts_p, faces_p, normals_p, _ = measure.marching_cubes(sdf_grid_pred, level=0.0, spacing=(spacing, spacing, spacing))
            verts_world_p = verts_p + np.array([-1.0, -1.0, -1.0])
            mesh_pred = trimesh.Trimesh(vertices=verts_world_p, faces=faces_p, process=False)
            if repair_mesh_flag:
                try:
                    import trimesh.repair as trepair
                    trepair.fill_holes(mesh_pred)
                    mesh_pred.remove_degenerate_faces()
                    mesh_pred.merge_vertices()
                except Exception:
                    pass
            pred_obj = f"{out_prefix}_sample{int(sample_idx)}_pred.obj"
            mesh_pred.export(pred_obj)
            print(f"[sample_dataset] Saved predicted OBJ to: {pred_obj}")
    except Exception as e:
        print(f"[sample_dataset] warning while building predicted OBJ: {e}")

    return out_file

def main():
    parser = argparse.ArgumentParser(description="Sample SDF from VAE prior and export a mesh (.obj)")
    parser.add_argument("--encoding-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64, help="VAE input / flattened latent dimension")
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--outfile", type=str, default="sampled_shape.obj")
    parser.add_argument("--prior-sigma", type=float, default=0.25)
    parser.add_argument("--ckpt", type=str, default="checkpoints_mod/mod_last.pth")
    parser.add_argument("--chunk-size", type=int, default=200000, help="chunk size for SDF evaluation")
    parser.add_argument("--scale", type=float, default=1.0, help="optional output mesh scale")
    parser.add_argument("--true-stats", default=True, help="compute and print statistics for ground-truth SDF data from the dataset")
    parser.add_argument("--pad-boundary", action="store_true", help="conservatively pad boundary voxels from adjacent interior slices to enclose the surface")
    parser.add_argument("--boundary-pad-value", type=float, default=1.0, help="minimum magnitude used when padding boundary voxels")
    parser.add_argument("--repair-mesh", action="store_true", help="attempt basic mesh repair (fill holes, remove degenerate faces) before export")
    parser.add_argument("--save-sample", action="store_true", help="save a dataset sample's GT and generated SDFs to .npz")
    parser.add_argument("--sample-idx", type=int, default=0, help="index of dataset sample to save")
    parser.add_argument("--sample-num-query", type=int, default=5000, help="num_query_points passed to VoxelSDFDataset when sampling")
    parser.add_argument("--sample-fixed-surface-size", type=int, default=5000, help="fixed_surface_points_size passed to VoxelSDFDataset when sampling")
    parser.add_argument("--sample-outprefix", type=str, default="sample", help="prefix for saved sample SDF .npz file")
    parser.add_argument("--sample-mesh-grid", type=int, default=64, help="grid resolution for predicted sample mesh export")
    parser.add_argument("--sample-mesh-chunk", type=int, default=20000, help="chunk size when evaluating dense grid for predicted mesh")
    parser.add_argument("--sample-repair-mesh", action="store_true", help="attempt mesh repair when exporting predicted sample OBJ")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build models (same construction as trainer.py)
    # Ensure VAE and SDF conditioning dims match: VAE.input_dim is the feature size
    # (encoding_dim) and VAE.latent_dim is the compressed z-size (latent_dim).
    # The SDF network is conditioned on the VAE decoder output (size encoding_dim),
    # so set sdf_network.latent_dim to encoding_dim.
    vae = ImprovedVAE(input_dim=args.encoding_dim, latent_dim=args.latent_dim, hidden_dim=512, num_layers=8).to(device)
    sdf_network = ImprovedSDFNetwork(input_dim=args.encoding_dim, latent_dim=args.encoding_dim, hidden_dim=128, output_dim=1, num_layers=8).to(device)
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

    # Optionally compute ground-truth dataset statistics for comparison
    if args.true_stats:
        try:
            print("Computing ground-truth SDF statistics from dataset (may take a little while)...")
            stats = compute_true_sdf_stats(root='.', name='sphere_complex', device='cpu')
            print("Ground-truth SDF stats:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"Warning: failed to compute true-data stats: {e}")

    sample_sdf_from_prior_and_save(modulation_module, device,
                                   filename=args.outfile,
                                   grid_resolution=int(args.grid),
                                   prior_sigma=float(args.prior_sigma),
                                   latent_dim=int(args.latent_dim),
                                   chunk_size=int(args.chunk_size),
                                   out_scale=float(args.scale),
                                   pad_boundary=bool(args.pad_boundary),
                                   boundary_pad_value=float(args.boundary_pad_value),
                                   repair_mesh=bool(args.repair_mesh))

    if args.save_sample:
        save_dataset_sample_sdfs(modulation_module, device,
                                 root='.', name='sphere_complex',
                                 sample_idx=int(args.sample_idx),
                                 num_query_points=int(args.sample_num_query),
                                 fixed_surface_points_size=int(args.sample_fixed_surface_size),
                                 out_prefix=str(args.sample_outprefix),
                                 pad_boundary=bool(args.pad_boundary),
                                 boundary_pad_value=float(args.boundary_pad_value),
                                 mesh_grid_resolution=int(args.sample_mesh_grid),
                                 mesh_chunk_size=int(args.sample_mesh_chunk),
                                 repair_mesh_flag=bool(args.sample_repair_mesh))

if __name__ == "__main__":
    main()
