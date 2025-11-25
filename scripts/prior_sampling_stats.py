#!/usr/bin/env python3
"""Sample several z from prior, decode via VAE.decoder, evaluate SDF on a 32^3 grid,
and report distribution of min/max/mean across samples.
"""
import os
import argparse
import numpy as np
import torch

from sample_sdf_obj import evaluate_sdf_in_chunks
from models import ImprovedVAE, ImprovedSDFNetwork, ModulationModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='checkpoints_mod/mod_last.pth')
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--grid', type=int, default=32)
    parser.add_argument('--prior-sigma', type=float, default=0.25)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print('Device:', device)

    # Model construction mirroring sample_sdf_obj
    encoding_dim = 128
    latent_dim = 1024
    vae = ImprovedVAE(input_dim=latent_dim, latent_dim=encoding_dim, hidden_dim=1024, num_layers=8).to(device)
    sdf_network = ImprovedSDFNetwork(input_dim=encoding_dim, latent_dim=latent_dim, hidden_dim=512, output_dim=1, num_layers=8).to(device)
    modulation = ModulationModule(vae, sdf_network).to(device)

    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        try:
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                modulation.load_state_dict(ckpt['model_state_dict'])
            else:
                modulation.load_state_dict(ckpt)
            print('Loaded checkpoint', args.ckpt)
        except Exception as e:
            print('Failed to load checkpoint:', e)
            return
    else:
        print('Checkpoint not found:', args.ckpt)
        return

    modulation.eval()

    # build grid
    lin = torch.linspace(-1.0, 1.0, args.grid, device=device)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing='ij'), dim=-1)  # [R,R,R,3]
    query_points = grid.reshape(-1, 3).unsqueeze(0)  # [1, N, 3]
    N = query_points.size(1)
    print(f'Grid {args.grid}^3 -> {N} points')

    mins = []
    maxs = []
    means = []

    # determine VAE latent dim (encoder output dim)
    vae_latent_dim = modulation.vae.fc_mu.out_features
    print('VAE latent dim (z size):', vae_latent_dim)

    for i in range(args.n):
        z = torch.randn(1, vae_latent_dim, device=device) * float(args.prior_sigma)
        with torch.no_grad():
            # use decoder to get conditioning vector
            x_cond = modulation.vae.decoder(z)
            sdf_flat = evaluate_sdf_in_chunks(modulation.sdf_network, query_points, x_cond, chunk_size=200_000)

        mins.append(float(np.min(sdf_flat)))
        maxs.append(float(np.max(sdf_flat)))
        means.append(float(np.mean(sdf_flat)))
        print(f'sample {i+1}/{args.n}: min={mins[-1]:.6f}, max={maxs[-1]:.6f}, mean={means[-1]:.6f}')

    def p(arr):
        a = np.array(arr)
        return {
            'min': float(a.min()),
            'max': float(a.max()),
            'p1': float(np.percentile(a, 1)),
            'p5': float(np.percentile(a, 5)),
            'p25': float(np.percentile(a, 25)),
            'p50': float(np.percentile(a, 50)),
            'p75': float(np.percentile(a, 75)),
            'p95': float(np.percentile(a, 95)),
            'p99': float(np.percentile(a, 99)),
        }

    print('Summary mins:', p(mins))
    print('Summary maxs:', p(maxs))
    print('Summary means:', p(means))


if __name__ == '__main__':
    main()
