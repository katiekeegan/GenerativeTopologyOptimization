#!/usr/bin/env python3
"""Train a DDPM in the VAE latent space.

This script mirrors the checkpoint-loading robustness from `sample_sdf_obj.py`.
It extracts latents z from the modulation module's VAE (using common output orders),
and trains a simple MLP diffusion model to predict noise in latent space.

Usage (example):
  python trainer_diffusion.py --ckpt checkpoints_mod/mod_last.pth --epochs 100 --batch 16
"""
import os
import argparse
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from models import ImprovedVAE, ImprovedSDFNetwork, ModulationModule
from dl4to.datasets import SELTODataset
from utils.preprocess_data import create_voxel_grids, VoxelSDFDataset, collate_fn

from trainer_diffusion import *


def extract_z_from_vae(modulation_module, point_clouds: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Robustly extract latent z (and mu/logvar if available) from the VAE.
    Returns (z, mu, logvar) where mu/logvar may be None if not available.
    - point_clouds: [B, ...] tensor
    """
    vae = modulation_module.vae
    # Ensure batch on device
    point_clouds = point_clouds.to(device)

    z = None
    mu = None
    logvar = None

    try:
        out = vae(point_clouds)
        # Common ImprovedVAE output per project: (x_recon, z, latent_pc, mu, logvar)
        if isinstance(out, tuple) or isinstance(out, list):
            if len(out) >= 2:
                # element 1 is often z
                z = out[1]
            if len(out) >= 5:
                mu = out[-2]
                logvar = out[-1]
        elif isinstance(out, torch.Tensor):
            # single tensor -> treat as z
            z = out
    except Exception:
        # try encoder API
        try:
            enc = vae.encode(point_clouds)
            # encode may return z directly or (mu, logvar)
            if isinstance(enc, tuple) or isinstance(enc, list):
                if len(enc) == 2:
                    mu, logvar = enc[0].to(device), enc[1].to(device)
                    std = (0.5 * logvar).exp()
                    eps = torch.randn_like(std)
                    z = mu + eps * std
                else:
                    z = enc[0]
            elif isinstance(enc, torch.Tensor):
                z = enc
        except Exception:
            raise RuntimeError("Failed to extract latents from VAE; check VAE API")

    # ensure z has shape [B, latent_dim]
    if z is None:
        raise RuntimeError("Could not obtain latent z from VAE outputs")
    if z.dim() == 3:
        # maybe [B, C, latent_dim] -> reduce/flatten per-sample mean
        z = z.mean(dim=1)

    return z, mu, logvar

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # Build modulation module (same as sample_sdf_obj/trainer.py)
    vae = ImprovedVAE(input_dim=args.encoding_dim, latent_dim=args.latent_dim, hidden_dim=512, num_layers=8).to(device)
    sdf_network = ImprovedSDFNetwork(input_dim=args.encoding_dim, latent_dim=args.encoding_dim, hidden_dim=128, output_dim=1, num_layers=8).to(device)
    modulation_module = ModulationModule(vae, sdf_network).to(device)

    # load checkpoint robustly
    if os.path.exists(args.modulation_ckpt):
        ckpt = torch.load(args.modulation_ckpt, map_location=device)
        try:
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                modulation_module.load_state_dict(ckpt["model_state_dict"])
            else:
                modulation_module.load_state_dict(ckpt)
            print(f"Loaded modulation module state_dict from {args.modulation_ckpt}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint into modulation_module: {e}. Proceeding with current init.")
    else:
        print(f"No checkpoint found at {args.modulation_ckpt}; proceeding with randomly initialized model (not recommended).")

    # dataset
    print("Loading SELTO dataset and creating voxel grids...")
    selto = SELTODataset(root='.', name=args.dataset_name, train=True)
    voxel_grids = create_voxel_grids(selto)
    F_list, Ω_design_list = create_problem_information_lists(selto)
    dataset = VoxelSDFDataset(voxel_grids, problem_information_list = [F_list, Ω_design_list], num_query_points=args.num_query_points, fixed_surface_points_size=args.fixed_surface_points_size, noise_std=0.0, device=device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # diffusion model
    diffusion = MLPDiffusionModel(latent_dim=args.latent_dim, hidden=args.hidden_dim, time_emb_dim=args.time_emb_dim).to(device)
    load_diffusion_checkpoint(diffusion, args.diffusion_ckpt, device)

    optimizer = optim.Adam(diffusion.parameters(), lr=args.lr)

def parse_args():
    parser = argparse.ArgumentParser(description='Train latent DDPM on VAE latents')
    parser.add_argument('--modulation-ckpt', type=str, default='checkpoints_mod/mod_last.pth')
    parser.add_argument('--diffusion-ckpt', type=str, default='checkpoints_diffusion/diffusion_epoch_1000.pth')
    parser.add_argument('--dataset-name', type=str, default='sphere_complex')
    parser.add_argument('--num-query-points', type=int, default=5000)
    parser.add_argument('--fixed-surface-points-size', type=int, default=5000)
    parser.add_argument('--encoding-dim', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--time-emb-dim', type=int, default=128)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--out-dir', type=str, default='checkpoints_diffusion')
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA even if available')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)