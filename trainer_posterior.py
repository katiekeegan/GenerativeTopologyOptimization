#!/usr/bin/env python3
"""
trainer_posterior.py
====================

This script implements variational posterior training in the latent space of a
Diffusion‑SDF model using a conditional RealNVP normalising flow and a
score‑based prior.  It is designed to follow the framework of Feng et al.
"Score‑Based Priors for Bayesian Inference"【895648704979186†L273-L317】 and thus
computes the log prior of a latent vector by integrating the probability flow
ODE (PF‑ODE) associated with a variance‑exploding score‑based diffusion model.

Overview
--------

1. **VAE and SDF:**  The training data are 3‑D signed distance fields (SDFs) of
   structural designs.  A pre‑trained Variational Autoencoder (VAE) encodes
   each SDF into a latent vector `z`.  The VAE decoder and an SDF network
   reconstruct SDFs from latent codes via the `ModulationModule` wrapper.
2. **Score‑Based Diffusion Prior:**  A score network trained via
   `trainer_diffusion.py` predicts the time‑dependent score (gradient of
   log density) of the latent variable distribution.  We treat this as our
   design prior.  To compute the log probability of a latent sample `z0`, we
   integrate the probability flow ODE

       d z_t/dt = -0.5 * σ(t)^2 * s_θ(z_t, t),

   where `σ(t)` is a variance‑exploding schedule and `s_θ` is the trained
   score network【895648704979186†L273-L317】.  The log density is then given by

       log p_θ(z0) = log p_T(z_T) - ∫₀¹ div(f_θ(z_t, t)) dt,

   where `p_T` is the Gaussian at terminal variance `σ_max` and the divergence
   term is approximated with Hutchinson’s estimator【895648704979186†L273-L317】.

3. **Conditional RealNVP Flow:**  We train a conditional RealNVP flow
   `q_φ(z | cond)` to approximate the posterior distribution over latent
   codes given problem information `cond` (e.g. forces and design masks).
   The flow transforms base noise `w ~ N(0,I)` into latent codes `z` using a
   sequence of affine coupling layers, each conditioned on an embedding of
   `cond` extracted by a small 3D convolutional encoder.
4. **Energy Functional:**  For each sample, we decode the latent code back to
   an SDF and compare it to the true SDF values at query points.  This
   reconstruction error defines a per‑sample energy `E(z)`; additional
   physics‑based terms (e.g. compliance) can be added here.  The total
   variational loss is

       L(φ) = E(z) - log p_θ(z) + log q_φ(z | cond),

   averaged over batches of samples from the flow.

This script trains only the RealNVP parameters and the conditioning encoder; the
VAE/SDF modules and the score network are loaded from existing checkpoints.

"""

import argparse
import math
import os
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import ImprovedVAE, ImprovedSDFNetwork, ModulationModule
from dl4to.datasets import SELTODataset
from utils.preprocess_data import (
    create_problem_information_lists,
    create_voxel_grids,
    VoxelSDFDataset,
    collate_fn,
)

from sample_sdf_obj import evaluate_sdf_in_chunks

from dl4to.problem import Problem
from dl4to.solution import Solution
# Pre-configure SciPy sparse solver before importing dl4to.pde to avoid
# AttributeError in certain SciPy versions when use_solver is called internally.
try:
    from scipy.sparse.linalg import use_solver as _scipy_use_solver
    # Initialize thread-local flags first, then set sorted-indices assumption.
    _scipy_use_solver(useUmfpack=False)
    _scipy_use_solver(assumeSortedIndices=True)
except Exception as _e:
    print(f"Warning: could not preconfigure SciPy solver: {_e}")
from dl4to.pde import FDM

# -----------------------------------------------------------------------------
# RealNVP implementation
# -----------------------------------------------------------------------------

def energy_functional(
    z: torch.Tensor,
    cond_grid: torch.Tensor,
    modulation_module=None,
    grid: int = 64,
    chunk_size: int = 4096,
    device: torch.device = torch.device("cpu"),
    eval_resolution: tuple = (39, 39, 21),
    problem_default = None,
    forces = None,
    Ω_design = None
) -> torch.Tensor:
    """Return aspect-ratio-preserving voxel density grids for each sample.

    Instead of computing energies, this function decodes z, evaluates SDF on an
    axis-wise grid with resolution `eval_resolution` in [-1,1]^3, and converts the
    SDF to a filled density grid (inside=1, outside=0). The output is a batched
    tensor of shape [B, D, H, W].

    Args:
        z: [B, latent_dim] latent codes
        cond_grid: [B, C_cond, D, H, W] conditioning grids (unused here)
        modulation_module: provides VAE decoder and SDF network
        grid: legacy cubic resolution (unused when eval_resolution provided)
        chunk_size: number of query points per chunk to avoid OOM
        device: torch device
        eval_resolution: (D, H, W) per-axis grid resolution

    Returns:
        energy: [B] tensor with per-sample energy (here: normalized max von Mises stress).
    """
    assert modulation_module is not None, "modulation_module must be provided"

    # Prepare high-resolution cubic query grid [1, N, 3] with resolution `grid`
    D_out, H_out, W_out = int(eval_resolution[0]), int(eval_resolution[1]), int(eval_resolution[2])
    lin = torch.linspace(-1.0, 1.0, int(grid), device=device)
    grid_coords = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), dim=-1)  # [G,G,G,3]
    query_points = grid_coords.reshape(-1, 3).unsqueeze(0)  # [1, N, 3], N=grid^3
    N = query_points.shape[1]

    # Decode latents to modulation vectors [B, encoding_dim]
    x_latent = modulation_module.vae.decoder(z)  # [B, encoding_dim]

    density_grids = []
    # Evaluate per-sample to keep memory reasonable
    for b in range(z.size(0)):
        x_latent_b = x_latent[b:b+1]  # [1, encoding_dim]
        # evaluate SDF in chunks; evaluator expects query_points [1,N,3]
        sdf_flat_b = evaluate_sdf_in_chunks(
            modulation_module.sdf_network,
            query_points,
            x_latent_b,
            chunk_size=chunk_size,
        )  # returns numpy array length N
        if sdf_flat_b.size != N:
            raise RuntimeError(f"Expected {N} SDF values but got {sdf_flat_b.size}")
        # Convert to torch and reshape to [G,G,G]
        G = int(grid)
        sdf_grid_b = torch.from_numpy(sdf_flat_b).to(device).reshape(G, G, G)
        # Resample SDF to target eval_resolution (D_out,H_out,W_out) via trilinear interpolation
        # Prepare as NCDHW: [1,1,G,G,G]
        sdf_ncdhw = sdf_grid_b.unsqueeze(0).unsqueeze(0)  # [1,1,G,G,G]
        sdf_resampled = torch.nn.functional.interpolate(
            sdf_ncdhw,
            size=(D_out, H_out, W_out),
            mode="trilinear",
            align_corners=True,
        ).squeeze(0).squeeze(0)  # [D_out,H_out,W_out]
        # Convert SDF to filled density: inside (sdf<=0) -> 1.0, outside -> 0.0
        density_b = (sdf_resampled <= 0.0).to(torch.float32)
        density_grids.append(density_b)

    batched_density_tensor = torch.stack(density_grids, dim=0)  # [B,D_out,H_out,W_out]
    von_Mises_stress_list = []
    for i in range(batched_density_tensor.size(0)):
        if problem_default is not None:
            # Work on a fresh clone to avoid in-place interference across samples
            problem_i = problem_default.clone()
            new_design = Ω_design[i].to(device=problem_i.device, dtype=problem_i.dtype)
            new_F = forces[i].to(device=problem_i.device, dtype=problem_i.dtype)
            problem_i.Ω_design.copy_(new_design)
            problem_i.F.copy_(new_F)
        else:
            raise RuntimeError("problem_default must be provided")
        problem_i.pde_solver = FDM()
        # Solution expects θ with shape [1, D, H, W]
        θ_i = batched_density_tensor[i].unsqueeze(0)
        solution_i = Solution(problem_i, θ_i)
        u_i, σ_i, σ_vm_i = solution_i.solve_pde()
        von_Mises_stress_i = σ_vm_i.max() / problem_i.σ_ys
        von_Mises_stress_list.append(von_Mises_stress_i)

    # Return per-sample energy as normalized max von Mises stress
    energy = torch.as_tensor(von_Mises_stress_list, device=device, dtype=z.dtype)
    return energy
    
class AffineCoupling(nn.Module):
    """Affine coupling layer as in RealNVP.

    Splits the input `x` into two parts along the last dimension.  The first
    part `x_a` is passed through unchanged; the second part `x_b` is affine
    transformed according to scale and translation parameters predicted by an
    MLP.  Both scale and translation are conditioned on an embedding of the
    problem information.
    """

    def __init__(self, in_dim: int, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.half = in_dim // 2
        # MLP to predict scale and translation; doubles hidden size for scale/shift
        self.net = nn.Sequential(
            nn.Linear(self.half + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (in_dim - self.half) * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward or inverse pass.

        Args:
            x: [B, in_dim]
            cond: [B, cond_dim]
            reverse: whether to perform the inverse transformation

        Returns:
            y: transformed tensor
            log_det_jac: log determinant of the Jacobian of this layer
        """
        x_a, x_b = x[:, : self.half], x[:, self.half :]
        # Condition on x_a concatenated with cond
        h = torch.cat([x_a, cond], dim=-1)
        # Predict scale and shift
        s_t = self.net(h)
        s, t = s_t.chunk(2, dim=-1)
        # Constrain scale to avoid numerical issues; tanh helps bound values
        s = torch.tanh(s)
        if reverse:
            # Inverse: x_b = (y_b - t) * exp(-s)
            y_b = (x_b - t) * torch.exp(-s)
            y_a = x_a
            log_det_jac = -torch.sum(s, dim=-1)
        else:
            # Forward: y_b = x_b * exp(s) + t
            y_b = x_b * torch.exp(s) + t
            y_a = x_a
            log_det_jac = torch.sum(s, dim=-1)
        y = torch.cat([y_a, y_b], dim=-1)
        return y, log_det_jac


class RealNVP(nn.Module):
    """A simple conditional RealNVP flow with alternating affine coupling layers and
    random permutations.  Conditioning is provided via an external embedding of
    the problem information (cond_emb)."""

    def __init__(self, latent_dim: int, hidden_dim: int, cond_dim: int, num_layers: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(AffineCoupling(latent_dim, hidden_dim, cond_dim))
            # After every coupling layer, apply a permutation by reversing the order
            # of dimensions to mix information.  This is implemented on the fly.

    def forward(self, w: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform base noise `w` into latent `z`.

        Args:
            w: [B, latent_dim] base samples ~ N(0, I)
            cond: [B, cond_dim] conditioning embedding

        Returns:
            z: latent codes
            log_det_sum: log determinant of the transformation
        """
        z = w
        log_det_sum = torch.zeros(w.size(0), device=w.device)
        for i, layer in enumerate(self.layers):
            z, log_det = layer(z, cond, reverse=False)
            log_det_sum += log_det
            # Permutation: reverse the dimensions except on the last layer
            if i + 1 < len(self.layers):
                z = z.flip(dims=[-1])
        return z, log_det_sum

    def inverse(self, z: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Invert the flow: map latent `z` back to base noise `w`.

        Args:
            z: [B, latent_dim]
            cond: [B, cond_dim]

        Returns:
            w: base samples
            log_det_sum: log determinant of the inverse transformation (negative of forward)
        """
        w = z
        log_det_sum = torch.zeros(z.size(0), device=z.device)
        for i in reversed(range(len(self.layers))):
            if i + 1 < len(self.layers):
                w = w.flip(dims=[-1])
            w, log_det = self.layers[i](w, cond, reverse=True)
            log_det_sum += log_det
        return w, log_det_sum


# -----------------------------------------------------------------------------
# Conditioning Encoder
# -----------------------------------------------------------------------------

class CondEncoder3D(nn.Module):
    """Encode the 3D conditioning grid into a small embedding vector.

    Given a tensor of shape [B, C_cond, D, H, W], the encoder applies a few
    convolutional layers followed by global average pooling to produce a vector
    of dimension `cond_emb_dim`.  This embedding is then passed to RealNVP
    coupling layers for conditioning.  Feel free to adjust the depth or hidden
    sizes as needed.
    """

    def __init__(self, in_channels: int, cond_emb_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(128, cond_emb_dim)

    def forward(self, cond_grid: torch.Tensor) -> torch.Tensor:
        # cond_grid: [B, C_cond, D, H, W]
        x = self.conv(cond_grid)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# -----------------------------------------------------------------------------
# Probability Flow ODE and log prior computation
# -----------------------------------------------------------------------------

def default_sigma_schedule(t: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """Return the VE sigma(t) schedule used for the probability flow ODE.

    We use the exponential interpolation between sigma_min and sigma_max as in
    score‑based diffusion models: σ(t) = σ_min * (σ_max / σ_min) ^ t.
    """
    return sigma_min * ((sigma_max / sigma_min) ** t)


def compute_log_prior_pf(
    z0: torch.Tensor,
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_steps: int = 50,
    sigma_min: float = 0.002,
    sigma_max: float = 50.0,
    noise_dim: int = None,
    rtol: float = 1e-5,
) -> torch.Tensor:
    """Compute log p_θ(z0) via the probability flow ODE using Euler integration.

    Args:
        z0: [B, d] latent vectors at time t=0
        score_fn: function returning the score s_θ(z_t,t) given z_t and continuous t
        num_steps: number of discrete integration steps between t=0 and t=1
        sigma_min, sigma_max: VE SDE bounds
        noise_dim: latent dimension (for Gaussian term); if None, derived from z0

    Returns:
        logp: [B] tensor of log probabilities under the score‑based prior

    Note:
        For efficiency, we perform a simple forward Euler integration.  More
        sophisticated solvers (e.g. RK45) can be implemented if desired.
    """
    device = z0.device
    B, d = z0.shape
    if noise_dim is None:
        noise_dim = d

    dt = 1.0 / num_steps
    # Initialize z_t and logp with zeros
    z_t = z0.clone().requires_grad_(True)
    logp = torch.zeros(B, device=device)
    # Precompute constant term for final Gaussian
    const_term = -0.5 * noise_dim * math.log(2 * math.pi)

    t = 0.0
    for i in range(num_steps):
        # Continuous time
        t_curr = torch.full((B, 1), t, device=device, dtype=z0.dtype)
        # Compute sigma(t)
        sigma_t = default_sigma_schedule(t_curr, sigma_min, sigma_max)
        # Score: [B, d]
        with torch.enable_grad():
            s_t = score_fn(z_t, t_curr.squeeze())
        # Drift for PF‑ODE
        drift = -0.5 * (sigma_t**2) * s_t
        # Hutchinson–Skilling estimator for divergence
        v = torch.randn_like(z_t)
        v_dot_drift = (v * drift).sum()
        grad = torch.autograd.grad(v_dot_drift, z_t, create_graph=True)[0]
        div_est = (grad * v).sum(dim=-1)  # [B]
        # Update log density: subtract divergence * dt
        logp = logp - div_est * dt
        # Euler update for z_t
        z_t = z_t + drift * dt
        t += dt
    # Final Gaussian log‑density at t=1 (sigma_max)
    logp = logp + const_term - 0.5 * ((z_t / sigma_max)**2).sum(dim=-1) - noise_dim * math.log(sigma_max)
    return logp


# -----------------------------------------------------------------------------
# Training function
# -----------------------------------------------------------------------------

def train(args) -> None:
    """Train the conditional RealNVP posterior sampler.

    This function orchestrates loading the pre‑trained VAE + SDF networks and
    score model, constructing the dataset with conditioning grids, building
    the conditioning encoder and RealNVP flow, and optimising the variational
    posterior using the PF‑ODE log prior.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Load modulation module (VAE + SDF network) and freeze its weights.
    # -------------------------------------------------------------------------
    vae = ImprovedVAE(input_dim=args.encoding_dim, latent_dim=args.latent_dim, hidden_dim=512, num_layers=8).to(device)
    sdf_network = ImprovedSDFNetwork(input_dim=args.encoding_dim, latent_dim=args.encoding_dim, hidden_dim=128, output_dim=1, num_layers=8).to(device)
    modulation_module = ModulationModule(vae, sdf_network).to(device)
    if os.path.exists(args.modulation_ckpt):
        mod_ckpt = torch.load(args.modulation_ckpt, map_location=device)
        try:
            if isinstance(mod_ckpt, dict) and "model_state_dict" in mod_ckpt:
                modulation_module.load_state_dict(mod_ckpt["model_state_dict"])
            else:
                modulation_module.load_state_dict(mod_ckpt)
            print(f"Loaded modulation module from {args.modulation_ckpt}")
        except Exception as e:
            print(f"Failed to load modulation checkpoint: {e}")
    else:
        print(f"Warning: modulation checkpoint {args.modulation_ckpt} not found. Using random init.")
    # Freeze VAE + SDF
    for p in modulation_module.parameters():
        p.requires_grad = False

    # -------------------------------------------------------------------------
    # Load pre‑trained score‑based diffusion model.  We reuse the MLP architecture
    # from trainer_diffusion.py.  The model takes a latent vector z and a time
    # index (encoded as continuous scalar in [0,1]) and predicts the VE score.
    # -------------------------------------------------------------------------
    from trainer_diffusion import MLPDiffusionModel

    # Initialize the score model; we'll re-instantiate with the correct cond_dim after
    # we infer the conditioning channels from the dataset below.
    score_model = MLPDiffusionModel(latent_dim=args.latent_dim, hidden=args.hidden_dim, time_emb_dim=args.time_emb_dim, cond_dim=0).to(device)
    # Re-instantiate score model with cond_dim=C_cond and reload checkpoint
    score_model = MLPDiffusionModel(latent_dim=args.latent_dim, hidden=args.hidden_dim, time_emb_dim=args.time_emb_dim, cond_dim=C_cond).to(device)
    if os.path.exists(args.diffusion_ckpt):
        diff_ckpt = torch.load(args.diffusion_ckpt, map_location=device)
        try:
            if isinstance(diff_ckpt, dict) and "model_state_dict" in diff_ckpt:
                score_model.load_state_dict(diff_ckpt["model_state_dict"])
            else:
                score_model.load_state_dict(diff_ckpt)
            print(f"Loaded diffusion model from {args.diffusion_ckpt}")
        except Exception as e:
            print(f"Failed to load diffusion checkpoint: {e}")
    else:
        print(f"Warning: diffusion checkpoint {args.diffusion_ckpt} not found. Using random init.")
    score_model.eval()
    for p in score_model.parameters():
        p.requires_grad = False

    def score_fn(z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the score s_θ(z_t, t) required for the PF‑ODE.

        Here `t` is a 1‑D tensor of size [B] representing the continuous time in
        [0,1].  The MLPDiffusionModel expects integer timesteps, but we treat
        the fractional value as a float; the time embedding will still map it
        appropriately.  We call the model with `cond=None` because the prior is
        unconditional.  The model was trained using a target of
            s_θ(z_t,t) ≈ -ε / σ(t)
        thus the output directly approximates the score.
        """
        # trainer_diffusion uses a VE-SDE with continuous time embedding via log sigma.
        # Compute log sigma(t) and pass as the time embedding; unconditional prior.
        sigma_t = default_sigma_schedule(t.unsqueeze(0), args.sigma_min, args.sigma_max).squeeze(0)
        t_embed = torch.log(sigma_t)
    # Condition on pooled FIRST-sample cond vector (broadcast to batch)
    cond_vec = cond_vec_first.expand(z_t.size(0), -1).contiguous()
    return score_model(z_t, t_embed, cond=cond_vec)

    # -------------------------------------------------------------------------
    # Data preparation: build voxel grids, problem information lists, and dataset.
    # -------------------------------------------------------------------------
    selto = SELTODataset(root=".", name=args.dataset_name, train=True)
    voxel_grids = create_voxel_grids(selto)
    problem, solution = selto[0]
    # Create lists of force tensors and design masks from SELTO; this function
    # should return two lists of length N: F_list and Omega_list
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
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Infer condition dimension from one sample; cond_grid is [B,C,D,H,W]
    sample = next(iter(dataloader))
    # Batch structure can be 4-tuple or 6-tuple depending on dataset settings
    if len(sample) == 6:
        _, _, _, cond_grid, _, _ = sample
    elif len(sample) == 4:
        _, _, _, cond_grid = sample
    else:
        raise RuntimeError("Unexpected batch structure from collate_fn")
    if isinstance(cond_grid, list):
        cond_grid = torch.stack(cond_grid, dim=0)
    C_cond = cond_grid.shape[1]
    # Pooled conditioning vector from FIRST dataset sample
    cond_first = cond_grid[0:1]
    cond_vec_first = cond_first.view(cond_first.size(1), -1).mean(dim=1, keepdim=True).t().to(device)

    # -------------------------------------------------------------------------
    # Build conditioning encoder and RealNVP flow
    # -------------------------------------------------------------------------
    cond_encoder = CondEncoder3D(in_channels=C_cond, cond_emb_dim=args.cond_emb_dim).to(device)
    flow = RealNVP(latent_dim=args.latent_dim, hidden_dim=args.flow_hidden_dim, cond_dim=args.cond_emb_dim, num_layers=args.flow_num_layers).to(device)

    # Optimiser over flow + cond encoder
    optimizer = optim.Adam(list(flow.parameters()) + list(cond_encoder.parameters()), lr=args.lr)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        flow.train()
        cond_encoder.train()
        epoch_loss = 0.0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            if len(batch) != 6:
                raise RuntimeError("collate_fn must return 6 elements: pc, qp, sdf_vals, cond, F, Ω_design")
            point_clouds, query_points, sdf_true, cond_grid, forces, Ω_design = batch
            # Stack point clouds to [B, N, 3]
            if isinstance(point_clouds, list):
                point_clouds = torch.stack(point_clouds, dim=0)
            else:
                point_clouds = point_clouds
            point_clouds = point_clouds.to(device)
            sdf_true = sdf_true.to(device)
            if isinstance(cond_grid, list):
                cond_grid = torch.stack(cond_grid, dim=0)
            cond_grid = cond_grid.to(device)

            # Extract latents from VAE; we ignore mu/logvar
            with torch.no_grad():
                z_gt = modulation_module.vae(point_clouds)
                if isinstance(z_gt, tuple) or isinstance(z_gt, list):
                    # according to ImprovedVAE outputs: (x_recon, z, latent_pc, mu, logvar)
                    z_gt = z_gt[1]
                if z_gt.dim() == 3:
                    z_gt = z_gt.mean(dim=1)
                z_gt = z_gt.detach()

            B = z_gt.size(0)
            # Compute cond embedding
            cond_emb = cond_encoder(cond_grid)
            # Sample base noise and transform to latent z
            w = torch.randn(B, args.latent_dim, device=device)
            z, log_det = flow(w, cond_emb)

            # Decode z to get predicted SDF at query points
            # Use modulation module: get decoder conditioning and call SDF network
            with torch.no_grad():
                # decode latents via VAE decoder; returns modulation vector x_mod of shape [B, encoding_dim]
                x_mod = modulation_module.vae.decoder(z)
                # Ensure query points are [B, N, 3]
                if isinstance(query_points, list):
                    query_pts = torch.stack(query_points, dim=0)
                else:
                    query_pts = query_points
                query_pts = query_pts.to(device)
                # Predict SDF; ImprovedSDFNetwork expects latent [B, encoding_dim] and expands internally
                sdf_pred = modulation_module.sdf_network(query_pts, x_mod)
                
            # Compute energy: mean squared error of SDF values
            # energy = F.mse_loss(sdf_pred, sdf_true, reduction="none").mean(dim=-1)  # [B]
            energy = energy_functional(z, cond_grid, modulation_module=modulation_module, device=device, problem_default=problem, forces=forces, Ω_design=Ω_design)
            # Compute log prior via PF‑ODE; detach z_gt?  We want gradient through z
            z_for_prior = z.detach().requires_grad_(True)
            logp = compute_log_prior_pf(z_for_prior, score_fn, num_steps=args.pf_steps, sigma_min=args.sigma_min, sigma_max=args.sigma_max)
            # Compute flow log prob (base density plus log det)
            logq = -0.5 * (w**2).sum(dim=-1) - 0.5 * args.latent_dim * math.log(2 * math.pi) + log_det
            # Loss per sample: energy - log prior + logq
            loss = energy - logp + logq
            # Mean over batch
            loss_mean = loss.mean()
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()
            epoch_loss += loss_mean.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss_mean.item():.4f}"})
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch}/{args.epochs} | Avg Loss: {avg_loss:.6f}")
        # Save checkpoint periodically
        if epoch % args.save_every == 0 or epoch == args.epochs:
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt_path = os.path.join(args.out_dir, f"posterior_epoch_{epoch}.pth")
            torch.save(
                {
                    "flow_state_dict": flow.state_dict(),
                    "cond_encoder_state_dict": cond_encoder.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RealNVP posterior with PF‑ODE prior")
    parser.add_argument("--modulation-ckpt", type=str, default="checkpoints_mod/mod_last.pth",
                        help="Path to modulation module checkpoint (VAE + SDF)")
    parser.add_argument("--diffusion-ckpt", type=str, default="checkpoints_diffusion/diffusion_epoch_1000.pth",
                        help="Path to pre‑trained diffusion score model checkpoint")
    parser.add_argument("--dataset-name", type=str, default="sphere_complex",
                        help="Name of SELTO dataset (e.g. sphere_complex, disc_simple)")
    parser.add_argument("--num-query-points", type=int, default=5000,
                        help="Number of query points sampled per shape for SDF supervision")
    parser.add_argument("--fixed-surface-points-size", type=int, default=5000,
                        help="Number of surface points fixed per shape")
    parser.add_argument("--encoding-dim", type=int, default=256,
                        help="Dimension of encoding used by VAE and SDF network")
    parser.add_argument("--latent-dim", type=int, default=64,
                        help="Dimension of VAE latent space")
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="Number of timesteps used in score training; used to discretise continuous time")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs for the posterior")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size for posterior training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for flow and cond encoder")
    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="Hidden dimension of the score network (unused here)")
    parser.add_argument("--time-emb-dim", type=int, default=128,
                        help="Dimension of time embedding for score network")
    parser.add_argument("--cond-emb-dim", type=int, default=128,
                        help="Dimension of conditioning embedding for RealNVP")
    parser.add_argument("--flow-hidden-dim", type=int, default=512,
                        help="Hidden dimension in RealNVP coupling nets")
    parser.add_argument("--flow-num-layers", type=int, default=6,
                        help="Number of affine coupling layers in RealNVP")
    parser.add_argument("--sigma-min", type=float, default=0.002,
                        help="Minimum sigma for VE SDE (PF‑ODE)")
    parser.add_argument("--sigma-max", type=float, default=50.0,
                        help="Maximum sigma for VE SDE (PF‑ODE)")
    parser.add_argument("--pf-steps", type=int, default=50,
                        help="Number of Euler steps for PF‑ODE integration")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--out-dir", type=str, default="checkpoints_posterior",
                        help="Output directory for posterior checkpoints")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)