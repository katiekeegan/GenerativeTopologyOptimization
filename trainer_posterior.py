#!/usr/bin/env python3
"""
trainer_posterior.py
====================

This script implements variational posterior training for latent shape codes
using a conditional normalizing flow (RealNVP) in the VAE latent space.  The
goal is to approximate a posterior distribution over latent codes conditioned
on problem information (e.g. loads and design masks) while encouraging the
generated shapes to be physically meaningful.  A physics-inspired energy
functional is computed on the fly in voxel space by comparing the predicted
signed distance field (SDF) with ground‐truth SDF values.  The variational
posterior is trained to minimise a free energy functional of the form

    F(q) = E_{z∼q}[E(z;cond) + ½‖z‖² − log|det J_φ|] ,

where E(z;cond) measures the mismatch between the predicted SDF (decoded
from z) and the observed SDF, ½‖z‖² is a standard normal prior on latent
codes, and log|det J_φ| is the log–Jacobian determinant of the flow.  The
learned flow thereby concentrates probability mass on latent codes that
produce low‐energy structures.

The script assumes that a VAE and SDF network (modulation module) have
already been trained (e.g. via Diffusion–SDF), and that a diffusion model
prior has been learned.  These networks are loaded and kept frozen; only
the flow and conditioning encoder are updated.  The conditioning grid is
built from SELTO problem information (load vectors and design masks) by
``VoxelSDFDataset``.

Note:  For simplicity and computational efficiency, the energy functional
implemented here is the mean–squared error (MSE) between predicted and
ground–truth SDF values on a set of query points.  This provides a
reasonable physics surrogate in voxel space without running full PDE
evaluations.  To incorporate true physical objectives (e.g. compliance),
replace the energy function with one that computes compliance via dl4to.

Example usage:

    python trainer_posterior.py \
        --modulation-ckpt checkpoints_mod/mod_last.pth \
        --diffusion-ckpt checkpoints_diffusion/diffusion_epoch_1000.pth \
        --epochs 50 --batch-size 4

"""

import os
import argparse
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import ImprovedVAE, ImprovedSDFNetwork, ModulationModule
from dl4to.datasets import SELTODataset
from utils.preprocess_data import create_voxel_grids, VoxelSDFDataset, collate_fn, create_problem_information_lists


# -----------------------------------------------------------------------------
# Conditional RealNVP implementation
# -----------------------------------------------------------------------------

class AffineCoupling(nn.Module):
    """A single affine coupling layer with optional conditioning.

    Splits the input latent vector into two halves along the feature dimension.
    One half is passed through an MLP to compute scale and translation vectors
    for the other half.  The conditioning embedding is concatenated to the
    active half before computing the transformation.
    """

    def __init__(self, dim: int, cond_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.half = dim // 2
        # MLP to produce scale and translation
        self.net = nn.Sequential(
            nn.Linear(self.half + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (dim - self.half) * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation.  Given input x (latent vector) and
        conditioning embedding cond, returns transformed output z and the
        log–determinant of the Jacobian.

        Args:
            x: [B, dim] latent vector
            cond: [B, cond_dim] conditioning embedding
        Returns:
            z: [B, dim] transformed latent
            log_det: [B] log–determinant of the transformation
        """
        x1 = x[:, :self.half]
        x2 = x[:, self.half:]
        # Concatenate conditioning to the active input
        h = torch.cat([x1, cond], dim=1)
        st = self.net(h)
        s, t = st[:, :x2.size(1)], st[:, x2.size(1):]
        # Use a small bound on scale to avoid numerical overflow
        s = torch.tanh(s)
        z1 = x1
        z2 = x2 * torch.exp(s) + t
        z = torch.cat([z1, z2], dim=1)
        log_det = s.sum(dim=1)
        return z, log_det

    def inverse(self, z: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation.  Given output z and conditioning, returns x.
        """
        z1 = z[:, :self.half]
        z2 = z[:, self.half:]
        h = torch.cat([z1, cond], dim=1)
        st = self.net(h)
        s, t = st[:, :z2.size(1)], st[:, z2.size(1):]
        s = torch.tanh(s)
        x1 = z1
        x2 = (z2 - t) * torch.exp(-s)
        x = torch.cat([x1, x2], dim=1)
        log_det = -s.sum(dim=1)
        return x, log_det


class RealNVP(nn.Module):
    """Conditional RealNVP composed of multiple affine coupling layers and
    optional dimension permutations between layers.

    Each coupling layer operates on half of the latent vector and uses a
    separate MLP to compute scale and translation.  A fixed random
    permutation is applied between layers to mix dimensions.
    """

    def __init__(self, latent_dim: int, cond_dim: int, num_couplings: int = 4, hidden_dim: int = 256):
        super().__init__()
        assert latent_dim % 2 == 0, "Latent dimension must be even for RealNVP"
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.num_couplings = num_couplings
        # Create couplings and random permutations
        self.couplings = nn.ModuleList()
        self.permutations = []
        for i in range(num_couplings):
            self.couplings.append(AffineCoupling(latent_dim, cond_dim, hidden_dim))
            # Create a random permutation of indices for this layer
            perm = torch.randperm(latent_dim)
            self.permutations.append(perm)

    def _permute(self, x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        return x[:, perm]

    def _unpermute(self, z: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(len(perm), device=perm.device)
        return z[:, inv_perm]

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the flow: maps base noise x to latent code z.
        Returns z and the accumulated log–determinant of Jacobians.

        Args:
            x: [B, latent_dim] base sample from N(0, I)
            cond: [B, cond_dim] conditioning embedding
        Returns:
            z: [B, latent_dim]
            log_det: [B] total log determinant
        """
        log_det_total = torch.zeros(x.size(0), device=x.device)
        h = x
        for coupling, perm in zip(self.couplings, self.permutations):
            h = self._permute(h, perm)
            h, log_det = coupling.forward(h, cond)
            log_det_total = log_det_total + log_det
        return h, log_det_total

    def inverse(self, z: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass through the flow: maps latent code z back to base noise x.
        Returns x and the accumulated inverse log–determinant.
        """
        log_det_total = torch.zeros(z.size(0), device=z.device)
        h = z
        # Apply couplings in reverse order
        for coupling, perm in reversed(list(zip(self.couplings, self.permutations))):
            h, log_det = coupling.inverse(h, cond)
            log_det_total = log_det_total + log_det
            h = self._unpermute(h, perm)
        return h, log_det_total

    def sample(self, num_samples: int, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw samples z ~ q_phi(z|cond) by sampling base noise and transforming.
        Returns z and log_det.
        """
        base = torch.randn(num_samples, self.latent_dim, device=cond.device)
        z, log_det = self.forward(base, cond)
        return z, log_det


class CondEncoder3D(nn.Module):
    """Small 3D convolutional encoder for conditioning grids.  Compresses
    input of shape [B, C, D, H, W] to a latent embedding [B, cond_dim]."""
    def __init__(self, in_channels: int, cond_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),  # [B, 64, 1, 1, 1]
        )
        self.fc = nn.Linear(64, cond_dim)

    def forward(self, cond_grid: torch.Tensor) -> torch.Tensor:
        x = self.conv(cond_grid)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# -----------------------------------------------------------------------------
# Training routine
# -----------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Load pretrained modulation module (VAE + SDF).  This is kept frozen.
    # -------------------------------------------------------------------------
    vae = ImprovedVAE(input_dim=args.encoding_dim, latent_dim=args.latent_dim,
                      hidden_dim=512, num_layers=8).to(device)
    sdf_network = ImprovedSDFNetwork(input_dim=args.encoding_dim, latent_dim=args.encoding_dim,
                                     hidden_dim=128, num_blocks=4, num_layers=4, output_dim=1).to(device)
    modulation_module = ModulationModule(vae, sdf_network).to(device)

    # Load checkpoint for modulation module
    if os.path.exists(args.modulation_ckpt):
        ckpt = torch.load(args.modulation_ckpt, map_location=device)
        try:
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                modulation_module.load_state_dict(ckpt["model_state_dict"])
            else:
                modulation_module.load_state_dict(ckpt)
            print(f"Loaded modulation module from {args.modulation_ckpt}")
        except Exception as e:
            print(f"Warning: failed to load modulation checkpoint: {e}")
    else:
        print(f"No modulation checkpoint found at {args.modulation_ckpt}; using random init.")

    # Freeze modulation module parameters
    for param in modulation_module.parameters():
        param.requires_grad = False

    # Optionally load a diffusion model; unused here except for prior analysis.
    if args.diffusion_ckpt and os.path.exists(args.diffusion_ckpt):
        # The diffusion model is not used directly in this training loop but
        # loading it here allows you to later compute log probabilities if
        # desired.  See trainer_diffusion.py for loading utilities.
        print(f"Diffusion checkpoint provided ({args.diffusion_ckpt}), but diffusion model is not used in this script.")

    # -------------------------------------------------------------------------
    # Prepare dataset and dataloader
    # -------------------------------------------------------------------------
    print("Loading SELTO dataset and computing voxel grids...")
    selto = SELTODataset(root='.', name=args.dataset_name, train=True)
    voxel_grids = create_voxel_grids(selto)
    # Build problem information lists for conditioning
    F_list, Ω_design_list = create_problem_information_lists(selto)
    problem_information_list = [F_list, Ω_design_list]
    dataset = VoxelSDFDataset(
        voxel_grids,
        problem_information_list=problem_information_list,
        num_query_points=args.num_query_points,
        fixed_surface_points_size=args.fixed_surface_points_size,
        noise_std=0.0,
        device=device,
        dataset='SELTO'
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Determine conditioning input channels from an example
    example = next(iter(dataloader))
    if len(example) == 4:
        _, _, _, cond_example = example
        C_cond = cond_example.size(1)
    else:
        raise RuntimeError("Dataset does not provide conditioning information; ensure problem_information_list is set.")

    # -------------------------------------------------------------------------
    # Set up conditional flow and conditioning encoder
    # -------------------------------------------------------------------------
    cond_encoder = CondEncoder3D(in_channels=C_cond, cond_dim=args.cond_dim).to(device)
    flow = RealNVP(latent_dim=args.latent_dim, cond_dim=args.cond_dim,
                   num_couplings=args.num_couplings, hidden_dim=args.flow_hidden_dim).to(device)
    params = list(cond_encoder.parameters()) + list(flow.parameters())
    optimizer = optim.Adam(params, lr=args.lr)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        cond_encoder.train()
        flow.train()
        epoch_loss = 0.0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            # Unpack batch: (surface_points, query_points, sdf_true, cond_grid)
            if len(batch) != 4:
                raise RuntimeError("Expected VoxelSDFDataset to return (surface_points, query_points, sdf, cond)")
            point_clouds, query_points, sdf_true, cond_grid = batch

            # Concatenate point_clouds into a batch tensor for VAE encoding
            if isinstance(point_clouds, list):
                pc_batch = torch.stack(point_clouds, dim=0).to(device)
            else:
                pc_batch = point_clouds.to(device)
            query_points = query_points.to(device)  # [B, N, 3]
            sdf_true = sdf_true.to(device)          # [B, N]
            cond_grid = cond_grid.to(device)        # [B, C_cond, D, H, W]

            # Encode conditioning grid to embedding
            cond_emb = cond_encoder(cond_grid)       # [B, cond_dim]

            # Sample base noise and transform through RealNVP
            base = torch.randn(pc_batch.size(0), args.latent_dim, device=device)
            z, log_det = flow(base, cond_emb)        # [B, latent_dim], [B]

            # Compute energy: decode latent to SDF predictions and compute MSE
            with torch.no_grad():
                # Use VAE decoder to reconstruct global feature from z
                x_recon = modulation_module.vae.decoder(z)
                # Evaluate SDF at query points
                sdf_pred = modulation_module.sdf_network(query_points, x_recon)  # [B, N]
            energy = F.mse_loss(sdf_pred, sdf_true, reduction='none').mean(dim=1)  # [B]

            # Prior term: standard normal prior on latent codes
            prior_term = 0.5 * torch.sum(z ** 2, dim=1)

            # Negative log determinant (since we minimise positive loss)
            loss = (energy + prior_term - log_det).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1
            global_step += 1
            if global_step % args.log_interval == 0:
                pbar.set_postfix({'loss': f"{loss.item():.6f}"})

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch} average loss: {avg_loss:.6f}")

        # Save checkpoint
        if epoch % args.save_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt = {
                'flow_state_dict': flow.state_dict(),
                'cond_encoder_state_dict': cond_encoder.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'args': vars(args),
            }
            fname = os.path.join(args.out_dir, f"posterior_epoch_{epoch}.pth")
            torch.save(ckpt, fname)
            print(f"Saved posterior checkpoint to {fname}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train conditional RealNVP posterior in latent space')
    parser.add_argument('--modulation-ckpt', type=str, default='checkpoints_mod/mod_last.pth',
                        help='path to pretrained modulation module (VAE + SDF) checkpoint')
    parser.add_argument('--diffusion-ckpt', type=str, default='',
                        help='(optional) diffusion prior checkpoint; unused in this script')
    parser.add_argument('--dataset-name', type=str, default='sphere_complex')
    parser.add_argument('--num-query-points', type=int, default=5000)
    parser.add_argument('--fixed-surface-points-size', type=int, default=5000)
    parser.add_argument('--encoding-dim', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--cond-dim', type=int, default=128, help='dimension of conditioning embedding')
    parser.add_argument('--num-couplings', type=int, default=4, help='number of coupling layers in RealNVP')
    parser.add_argument('--flow-hidden-dim', type=int, default=256, help='hidden dimension in coupling MLPs')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--out-dir', type=str, default='checkpoints_posterior')
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA even if available')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)