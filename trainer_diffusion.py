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


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule as in Nichol & Dhariwal implementation.
    Returns betas tensor shape [timesteps]."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


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


def sinusoidal_timestep_embedding(timesteps, dim: int):
    # Ensure all tensors are created on the same device as `timesteps` to avoid device mismatch
    device = timesteps.device
    half = dim // 2
    # build frequency bands on the correct device
    log_max = math.log(10000.0)
    inv_freq = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -(log_max / (half - 1)))
    t = timesteps.float().unsqueeze(1).to(device)
    emb = t * inv_freq.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class MLPDiffusionModel(nn.Module):
    """Simple MLP that predicts noise in latent space, conditioned on timestep embedding."""
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
        # x: [B, latent_dim], t: [B] ints
        t_emb = sinusoidal_timestep_embedding(t, self.time_emb_dim)
        t_emb = t_emb.to(x.device)
        t_emb = self.time_mlp(t_emb)
        h = torch.cat([x, t_emb], dim=1)
        return self.net(h)


def forward_process(z0: torch.Tensor, t: torch.Tensor, betas: torch.Tensor, alphas_cumprod: torch.Tensor):
    """Add noise to z0 at timestep t. Returns z_t and the noise epsilon used."""
    device = z0.device
    betas = betas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # gather per-sample
    sqrt_ac = sqrt_alphas_cumprod[t].unsqueeze(1)
    sqrt_om = sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)

    noise = torch.randn_like(z0)
    z_t = sqrt_ac * z0 + sqrt_om * noise
    return z_t, noise


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # Build modulation module (same as sample_sdf_obj/trainer.py)
    vae = ImprovedVAE(input_dim=args.encoding_dim, latent_dim=args.latent_dim, hidden_dim=512, num_layers=8).to(device)
    sdf_network = ImprovedSDFNetwork(input_dim=args.encoding_dim, latent_dim=args.encoding_dim, hidden_dim=128, output_dim=1, num_layers=8).to(device)
    modulation_module = ModulationModule(vae, sdf_network).to(device)

    # load checkpoint robustly
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

    # dataset
    print("Loading SELTO dataset and creating voxel grids...")
    selto = SELTODataset(root='.', name=args.dataset_name, train=True)
    voxel_grids = create_voxel_grids(selto)
    dataset = VoxelSDFDataset(voxel_grids, num_query_points=args.num_query_points, fixed_surface_points_size=args.fixed_surface_points_size, noise_std=0.0, device=device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # diffusion model
    diffusion = MLPDiffusionModel(latent_dim=args.latent_dim, hidden=args.hidden_dim, time_emb_dim=args.time_emb_dim).to(device)
    optimizer = optim.Adam(diffusion.parameters(), lr=args.lr)

    # schedules
    betas = cosine_beta_schedule(args.timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    global_step = 0
    for epoch in range(args.epochs):
        diffusion.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        for batch in pbar:
            # batch: (point_clouds:list, query_points, sdf) per VoxelSDFDataset collate
            try:
                point_clouds = batch[0]
            except Exception:
                raise RuntimeError("Unexpected batch format from dataset; expected collate_fn output")

            # point_clouds might be list of tensors per sample
            if isinstance(point_clouds, list):
                try:
                    pc_batch = torch.stack(point_clouds, dim=0).to(device)
                except Exception:
                    # fallback: if already batched tensor
                    pc_batch = point_clouds[0].unsqueeze(0).to(device)
            else:
                pc_batch = point_clouds.to(device)

            # extract z from VAE
            try:
                z, mu, logvar = extract_z_from_vae(modulation_module, pc_batch, device)
            except Exception as e:
                print(f"Error extracting z from VAE: {e}")
                continue

            # ensure z is [B, latent_dim]
            z = z.view(z.size(0), -1)

            # sample timesteps per sample
            t = torch.randint(0, args.timesteps, (z.size(0),), device=device)

            z_t, noise = forward_process(z, t, betas, alphas_cumprod)

            optimizer.zero_grad()
            predicted = diffusion(z_t, t)
            loss = F.mse_loss(predicted, noise)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            global_step += 1
            if global_step % args.log_interval == 0:
                pbar.set_postfix({'loss': f"{loss.item():.6f}"})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} avg diffusion loss: {avg_loss:.6f}")

        # checkpoint
        if (epoch + 1) % args.save_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt_model = {
                'model_state_dict': diffusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
            }
            fname = os.path.join(args.out_dir, f"diffusion_epoch_{epoch+1}.pth")
            torch.save(ckpt_model, fname)
            print(f"Saved diffusion checkpoint: {fname}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train latent DDPM on VAE latents')
    parser.add_argument('--ckpt', type=str, default='checkpoints_mod/mod_last.pth')
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
