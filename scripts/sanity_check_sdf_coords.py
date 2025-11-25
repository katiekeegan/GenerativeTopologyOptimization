#!/usr/bin/env python3
"""
Sanity-check script for SDF coordinate/axis/anchor consistency.

Runs three tests:
 1) marching_cubes axis-order probe
 2) torch.nn.functional.grid_sample normalized-coordinate probe
 3) VoxelSDFDataset round-trip on a synthetic sphere voxel grid

Run from the repo root: python scripts/sanity_check_sdf_coords.py
"""
import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure

from utils.preprocess_data import VoxelSDFDataset


def test_marching_cubes_axis():
    print("\n[TEST 1] marching_cubes axis-order probe")
    D,H,W = 6,8,10
    vol = np.zeros((D,H,W), dtype=np.uint8)
    vol[0,0,0] = 1
    verts, faces, normals, _ = measure.marching_cubes(vol, level=0.5)
    print("verts shape:", verts.shape)
    print("verts min per axis:", verts.min(axis=0))
    print("verts max per axis:", verts.max(axis=0))
    print("sample verts:\n", verts[:min(8, len(verts))])
    return verts


def test_grid_sample_probe():
    print("\n[TEST 2] grid_sample normalized coordinate probe")
    D,H,W = 5,7,9
    sdf = torch.zeros(1, 1, D, H, W, dtype=torch.float32)
    sdf[0,0,0,0,0] = 3.14
    # coords: [N, D_out, H_out, W_out, 3]
    coords = torch.tensor([[[[[ -1.0, -1.0, -1.0 ]]]]], dtype=torch.float32)
    sampled = F.grid_sample(sdf, coords, align_corners=True)
    val = sampled.squeeze().item()
    print("Sampled RAW grid value at normalized coord (-1,-1,-1):", val)
    print("  (This is a raw value in voxel units from the sdf grid before any dataset normalization.)")
    return val


def test_dataset_roundtrip():
    print("\n[TEST 3] VoxelSDFDataset round-trip on synthetic sphere voxel grid")
    D,H,W = 21,21,21
    cx,cy,cz = (W-1)/2.0, (H-1)/2.0, (D-1)/2.0
    r = 5.0
    vol = np.zeros((D,H,W), dtype=np.uint8)
    for z in range(D):
        for y in range(H):
            for x in range(W):
                if (x-cx)**2 + (y-cy)**2 + (z-cz)**2 <= r*r:
                    vol[z,y,x] = 1
    vg = torch.tensor(vol, dtype=torch.float32)
    # explicit anchor='min' to match sample_sdf_obj mapping of index 0 -> -1
    dataset = VoxelSDFDataset([vg], num_query_points=1000, fixed_surface_points_size=500, noise_std=0.0, device='cpu', anchor='min')
    sp, qp, sdf = dataset[0]
    print("surface points shape:", sp.shape)
    print("query points shape:", qp.shape)
    print("sdf stats: min={}, max={}, mean={}".format(float(sdf.min()), float(sdf.max()), float(sdf.mean())))
    # sample grid_sample on the underlying sdf grid directly at normalized coords [-1,-1,-1] to compare
    sdf_grid = dataset.sdf_grids[0].unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    coords = torch.tensor([[[[[ -1.0, -1.0, -1.0 ]]]]], dtype=torch.float32)
    raw_val = F.grid_sample(sdf_grid, coords, align_corners=True).squeeze().item()
    normalized_from_raw = raw_val / dataset.sdf_scale
    print("Direct RAW grid_sample on sdf_grid at (-1,-1,-1) (voxel units):", raw_val)
    print(f"Normalized_from_raw = raw_val / dataset.sdf_scale ({dataset.sdf_scale:.6f}) -> {normalized_from_raw:.6f}")
    print("Dataset-returned normalized SDF stats (from sampled query points):",
        f"min={float(sdf.min()):.6f}, max={float(sdf.max()):.6f}, mean={float(sdf.mean()):.6f}")
    print("  Note: dataset returns SDFs already normalized and clamped to [-1,1].")
    return sp, qp, sdf, raw_val


def main():
    verts = test_marching_cubes_axis()
    val = test_grid_sample_probe()
    sp, qp, sdf, val2 = test_dataset_roundtrip()
    print("\nSanity checks completed. Inspect outputs above for axis/anchor mismatches.")


if __name__ == '__main__':
    main()
