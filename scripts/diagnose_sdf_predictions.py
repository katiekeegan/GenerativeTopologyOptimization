#!/usr/bin/env python3
"""Diagnostic: compare SDF predictions under different conditioning inputs.

Loads modulation checkpoint, grabs one dataset sample, and evaluates:
 - model conditioned on x_recon produced by the VAE from the sample's surface points
 - model conditioned on the raw z produced by the VAE encoder
 - model conditioned on decoder(z)

Prints min/max/mean/std/percentiles for each prediction and differences vs ground-truth.
"""
import os
import argparse
import numpy as np
import torch

from models import ImprovedVAE, ImprovedSDFNetwork, ModulationModule
from utils.preprocess_data import create_voxel_grids, VoxelSDFDataset
from dl4to.datasets import SELTODataset


def stats(x):
    a = x.flatten()
    return {
        'min': float(a.min()),
        'max': float(a.max()),
        'mean': float(a.mean()),
        'std': float(a.std()),
        'p10': float(np.percentile(a, 10)),
        'p50': float(np.percentile(a, 50)),
        'p90': float(np.percentile(a, 90)),
    }


def print_stats(prefix, arr):
    s = stats(arr)
    print(f"{prefix}: min={s['min']:.6f}, max={s['max']:.6f}, mean={s['mean']:.6f}, std={s['std']:.6f}, p10={s['p10']:.6f}, p50={s['p50']:.6f}, p90={s['p90']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='checkpoints_mod/mod_last.pth')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sample-idx', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # build models (match `sample_sdf_obj.py` construction and the trained checkpoint)
    # sample_sdf_obj used: ImprovedVAE(input_dim=latent_dim, latent_dim=encoding_dim, hidden_dim=1024, num_layers=8)
    # and ImprovedSDFNetwork(input_dim=encoding_dim, latent_dim=latent_dim, hidden_dim=512, output_dim=1, num_layers=8)
    encoding_dim = 128
    latent_dim = 1024
    vae = ImprovedVAE(input_dim=latent_dim, latent_dim=encoding_dim, hidden_dim=1024, num_layers=8).to(device)
    sdf_network = ImprovedSDFNetwork(input_dim=encoding_dim, latent_dim=latent_dim, hidden_dim=512, output_dim=1, num_layers=8).to(device)
    modulation = ModulationModule(vae, sdf_network).to(device)

    # load checkpoint
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        try:
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                modulation.load_state_dict(ckpt['model_state_dict'])
            else:
                modulation.load_state_dict(ckpt)
            print(f"Loaded checkpoint: {args.ckpt}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    else:
        raise RuntimeError(f"Checkpoint not found: {args.ckpt}")

    modulation.eval()

    # load dataset voxel grids (use SELTO default name)
    selto = SELTODataset(root='.', name='sphere_complex', train=True)
    voxel_grids = create_voxel_grids(selto)
    dataset = VoxelSDFDataset(voxel_grids, num_query_points=2000, fixed_surface_points_size=2000, noise_std=0.0, device=device)

    # take one sample
    surf_pts, query_pts, sdf_gt = dataset[args.sample_idx]
    # ensure shapes: surf_pts [N,3], query_pts [num_q,3], sdf_gt [num_q]
    print(f"Sample {args.sample_idx} shapes: surf_pts={tuple(surf_pts.shape)}, query_pts={tuple(query_pts.shape)}, sdf_gt={tuple(sdf_gt.shape)}")
    # quick checks on query points
    qp_np = query_pts.cpu().numpy()
    print("query_pts[0:5]:", qp_np[:5])
    print("query_pts per-axis std:", np.std(qp_np, axis=0).tolist())

    # prepare batch dims
    surf_b = surf_pts.unsqueeze(0).to(device)
    query_b = query_pts.unsqueeze(0).to(device)
    sdf_b = sdf_gt.unsqueeze(0).to(device)

    # get VAE outputs for the sample
    with torch.no_grad():
        x_recon, z_enc, latent_pc, mu, logvar = modulation.vae(surf_b)

        # predictions conditioned on x_recon (should match SDF latent_dim)
        pred_xrecon = modulation.sdf_network(query_b, x_recon).cpu().numpy()
        print('x_recon sample stats: mean=', float(x_recon.mean().cpu()), 'std=', float(x_recon.std().cpu()))
        print('x_recon[0:10]:', x_recon.flatten()[:10].cpu().numpy().tolist())

        # try predictions conditioned on raw z_enc only if sizes match
        pred_z = None
        if z_enc.shape[-1] == latent_dim:
            pred_z = modulation.sdf_network(query_b, z_enc).cpu().numpy()
        else:
            print(f"Skipping pred_z: encoder z size={z_enc.shape[-1]} != sdf latent_dim={latent_dim}")

        # predictions conditioned on decoder(z_enc) (decoder -> x_recon-like)
        try:
            x_from_decoder = modulation.vae.decoder(z_enc)
            pred_decoder = modulation.sdf_network(query_b, x_from_decoder).cpu().numpy()
        except Exception:
            pred_decoder = None

        gt = sdf_b.cpu().numpy()

    print_stats('GT (normalized)', gt)
    print_stats('Pred conditioned on x_recon', pred_xrecon)
    print('first 10 preds (x_recon):', pred_xrecon.flatten()[:10].tolist())
    if pred_z is not None:
        print_stats('Pred conditioned on z_enc', pred_z)
        print('first 10 preds (z_enc):', pred_z.flatten()[:10].tolist())
    else:
        print('(pred_z skipped due to latent size mismatch)')
    if pred_decoder is not None:
        print_stats('Pred conditioned on decoder(z_enc)', pred_decoder)
        print('first 10 preds (decoder):', pred_decoder.flatten()[:10].tolist())

    # print differences (only for existing preds)
    if pred_xrecon is not None:
        d = pred_xrecon - gt
        print_stats('Diff (pred_xrecon - gt)', d)
    if pred_z is not None:
        d = pred_z - gt
        print_stats('Diff (pred_z - gt)', d)
    if pred_decoder is not None:
        d = pred_decoder - gt
        print_stats('Diff (pred_decoder - gt)', d)


if __name__ == '__main__':
    main()
