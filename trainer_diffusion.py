#!/usr/bin/env python3
"""Train a score-based diffusion model in the VAE latent space.

This script mirrors the checkpoint-loading robustness from `sample_sdf_obj.py`.
It extracts latents z from the modulation module's VAE (using common output orders),
and trains a simple MLP diffusion model to predict noise in latent space.

Usage (example):
  python trainer_diffusion.py --ckpt checkpoints_mod/mod_last.pth --epochs 100 --batch 16
"""
import os
import argparse
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from models import ImprovedVAE, ImprovedSDFNetwork, ModulationModule
from dl4to.datasets import SELTODataset
from utils.preprocess_data import create_voxel_grids, VoxelSDFDataset, collate_fn, create_problem_information_lists


def ve_sigma_schedule(t_min: float, t_max: float):
    """Return helpers for a Variance-Exploding (VE) SDE noise schedule.
    We parameterize sigma(t) = sigma_min * (sigma_max/sigma_min) ** t, with t in [0,1].
    """
    sigma_min = t_min
    sigma_max = t_max

    def sigma_from_t(t: torch.Tensor) -> torch.Tensor:
        # t ∈ [0,1], sigma(t) = sigma_min * exp(log(sigma_max/sigma_min) * t)
        return sigma_min * torch.exp(torch.log(torch.tensor(sigma_max / sigma_min, device=t.device)) * t)

    def embed_t_for_time(t: torch.Tensor) -> torch.Tensor:
        # Use log sigma as the embedding input to sinusoidal_timestep_embedding
        sig = sigma_from_t(t)
        log_sig = torch.log(sig)
        return log_sig

    return sigma_from_t, embed_t_for_time


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
    """Simple MLP that predicts noise in latent space.
    Supports optional conditioning via a spatial grid cond: [B, C_cond, D, H, W].
    The conditioning is pooled to a vector and concatenated to inputs.
    """
    def __init__(self, latent_dim: int, hidden: int = 512, time_emb_dim: int = 128, cond_dim: Optional[int] = None):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        in_dim = latent_dim + time_emb_dim + (cond_dim or 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, cond: torch.Tensor | None = None):
        # x: [B, latent_dim], t_embed: [B] real-valued (e.g., log-sigma), cond: [B, C_cond, D, H, W] or None
        t_emb = sinusoidal_timestep_embedding(t_embed, self.time_emb_dim)
        t_emb = t_emb.to(x.device)
        t_emb = self.time_mlp(t_emb)

        if cond is not None:
            # Global average pool over spatial dims -> [B, C_cond]
            if cond.dim() == 5:
                cond_vec = cond.mean(dim=(-1, -2, -3))
            elif cond.dim() == 2:
                cond_vec = cond
            else:
                # Fallback: flatten last dims to vector
                cond_vec = cond.view(cond.size(0), -1)
            cond_vec = cond_vec.to(x.device)
            h = torch.cat([x, t_emb, cond_vec], dim=1)
        else:
            # If the network was initialized with a cond_dim, but no cond is provided,
            # pad zeros so the input dimension matches.
            if self.cond_dim is not None and self.cond_dim > 0:
                cond_vec = torch.zeros(x.size(0), self.cond_dim, device=x.device, dtype=x.dtype)
                h = torch.cat([x, t_emb, cond_vec], dim=1)
            else:
                h = torch.cat([x, t_emb], dim=1)

        return self.net(h)


def ve_forward_noising(z0: torch.Tensor, sigma: torch.Tensor):
    """Add VE-SDE noise: z_t = z0 + sigma * noise.
    Returns z_t and the noise epsilon used."""
    noise = torch.randn_like(z0)
    z_t = z0 + sigma.unsqueeze(1) * noise
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
    if args.cond:
        F_list, Ω_design_list = create_problem_information_lists(selto)
        dataset = VoxelSDFDataset(voxel_grids, problem_information_list = [F_list, Ω_design_list], num_query_points=args.num_query_points, fixed_surface_points_size=args.fixed_surface_points_size, noise_std=0.0, device=device)
    else:
        dataset = VoxelSDFDataset(voxel_grids, num_query_points=args.num_query_points, fixed_surface_points_size=args.fixed_surface_points_size, noise_std=0.0, device=device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # diffusion model (score-based)
    # If conditioning, infer cond_dim from dataset's conditioning channels (C_cond=8 in current SELTO processing)
    cond_dim = 8 if args.cond else None
    diffusion = MLPDiffusionModel(latent_dim=args.latent_dim, hidden=args.hidden_dim, time_emb_dim=args.time_emb_dim, cond_dim=cond_dim).to(device)
    optimizer = optim.Adam(diffusion.parameters(), lr=args.lr)

    # VE schedule helpers
    sigma_from_t, embed_t_for_time = ve_sigma_schedule(args.sigma_min, args.sigma_max)

    global_step = 0
    for epoch in range(args.epochs):
        diffusion.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        for batch in pbar:
            # batch: (point_clouds:list, query_points, sdf) per VoxelSDFDataset collate
            try:
                point_clouds = batch[0]
                # optional conditioning tensor from collate_fn when args.cond
                batch_cond = batch[3] if (args.cond and len(batch) >= 4) else None
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

            # sample continuous times t ∈ [0,1]
            t = torch.rand(z.size(0), device=device)
            sigma = sigma_from_t(t)
            z_t, noise = ve_forward_noising(z, sigma)

            optimizer.zero_grad()
            # Use log-sigma embedding for time
            t_embed = embed_t_for_time(t)
            predicted_score = diffusion(z_t, t_embed, cond=batch_cond)
            # Target score for VE: s* = - noise / sigma
            target_score = - noise / sigma.unsqueeze(1)
            # Optional weighting lambda(t): sigma^2 (to balance scales). Here we use simple MSE on score.
            loss = F.mse_loss(predicted_score, target_score)
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
    # score-based (VE) schedule params
    parser.add_argument('--sigma-min', type=float, default=0.01)
    parser.add_argument('--sigma-max', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--time-emb-dim', type=int, default=128)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--out-dir', type=str, default='checkpoints_diffusion')
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA even if available')
    parser.add_argument('--cond', action='store_true', help='use conditional diffusion')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    train(args)
