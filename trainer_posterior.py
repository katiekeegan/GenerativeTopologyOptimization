#!/usr/bin/env python3
"""
trainer_posterior.py
====================

This script implements variational posterior training in the latent space of a
Diffusion‑SDF model using a single-observation RealNVP normalising flow and a
score‑based prior. It is designed to follow the framework of Feng et al.
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

      d z_t/dt = -0.5 * g(t)^2 * s_θ(z_t, t),

   where `σ(t)` is a variance‑exploding schedule, `g(t)^2 = dσ(t)^2/dt`,
   and `s_θ` is the trained score network【895648704979186†L273-L317】.
   The log density is then given by

       log p_θ(z0) = log p_T(z_T) - ∫₀¹ div(f_θ(z_t, t)) dt,

   where `p_T` is the Gaussian at terminal variance `σ_max` and the divergence
   term is approximated with Hutchinson’s estimator【895648704979186†L273-L317】.

3. **RealNVP Flow:**  We train a RealNVP flow `q_φ(z)` for one fixed problem
   observation. The flow transforms base noise `w ~ N(0,I)` into latent codes
   `z` using a sequence of affine coupling layers.
4. **Energy Functional:**  For each sample, we decode the latent code back to
   an SDF and compare it to the true SDF values at query points.  This
   reconstruction error defines a per‑sample energy `E(z)`; additional
   physics‑based terms (e.g. compliance) can be added here.  The total
   variational loss is

      L(φ) = E(z) - log p_θ(z) + log q_φ(z),

   averaged over batches of samples from the flow.

This script trains only the RealNVP parameters; the VAE/SDF modules and the
score network are loaded from existing checkpoints.

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
    grid: int = 32,
    chunk_size: int = 4096,
    device: torch.device = torch.device("cpu"),
    eval_resolution: tuple = (39, 39, 21),
    problem_default = None,
    forces = None,
    Ω_design = None,
    pde_solver=None,
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
            # To avoid device mismatches inside dl4to.pde, keep all PDE tensors on CPU
            cpu = torch.device("cpu")
            new_design = Ω_design[i].to(device=cpu, dtype=problem_i.dtype)
            new_F = forces[i].to(device=cpu, dtype=problem_i.dtype)
            problem_i.Ω_design.copy_(new_design)
            problem_i.F.copy_(new_F)
        else:
            raise RuntimeError("problem_default must be provided")
        # Reuse a preassembled PDE solver if provided
        if pde_solver is not None:
            problem_i.pde_solver = pde_solver
        else:
            problem_i.pde_solver = FDM()
        # Solution expects θ with shape [1, D, H, W]
        θ_i = batched_density_tensor[i].unsqueeze(0).to(cpu)
        solution_i = Solution(problem_i, θ_i)
        u_i, σ_i, σ_vm_i = solution_i.solve_pde()
        von_Mises_stress_i = σ_vm_i.max() / problem_i.σ_ys
        von_Mises_stress_list.append(von_Mises_stress_i)

    # Return per-sample energy as normalized max von Mises stress
    energy = torch.as_tensor(von_Mises_stress_list, device=device, dtype=z.dtype)
    return energy
    
class AffineCoupling(nn.Module):
    """Affine coupling layer as in RealNVP (unconditional).

    Splits the input `x` into two parts along the last dimension.  The first
    part `x_a` is passed through unchanged; the second part `x_b` is affine
    transformed according to scale and translation parameters predicted by an
    MLP from `x_a` only (no conditioning).
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.half = in_dim // 2
        self.net = nn.Sequential(
            nn.Linear(self.half, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (in_dim - self.half) * 2),
        )

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x_a, x_b = x[:, : self.half], x[:, self.half :]
        h = x_a
        s_t = self.net(h)
        s, t = s_t.chunk(2, dim=-1)
        s = torch.tanh(s)
        if reverse:
            y_b = (x_b - t) * torch.exp(-s)
            y_a = x_a
            log_det_jac = -torch.sum(s, dim=-1)
        else:
            y_b = x_b * torch.exp(s) + t
            y_a = x_a
            log_det_jac = torch.sum(s, dim=-1)
        y = torch.cat([y_a, y_b], dim=-1)
        return y, log_det_jac


class RealNVP(nn.Module):
    """Unconditional RealNVP flow with alternating affine coupling layers and
    random permutations (dimension reversal)."""

    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(AffineCoupling(latent_dim, hidden_dim))

    def forward(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = w
        log_det_sum = torch.zeros(w.size(0), device=w.device)
        for i, layer in enumerate(self.layers):
            z, log_det = layer(z, reverse=False)
            log_det_sum += log_det
            if i + 1 < len(self.layers):
                z = z.flip(dims=[-1])
        return z, log_det_sum

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = z
        log_det_sum = torch.zeros(z.size(0), device=z.device)
        for i in reversed(range(len(self.layers))):
            if i + 1 < len(self.layers):
                w = w.flip(dims=[-1])
            w, log_det = self.layers[i](w, reverse=True)
            log_det_sum += log_det
        return w, log_det_sum


def standard_normal_logprob(w: torch.Tensor) -> torch.Tensor:
    return -0.5 * (w**2).sum(dim=-1) - 0.5 * w.size(-1) * math.log(2 * math.pi)


def flow_log_prob(flow: RealNVP, z: torch.Tensor) -> torch.Tensor:
    """Evaluate log q_phi(z) with z treated as the density-query point."""
    w, inverse_log_det = flow.inverse(z)
    return standard_normal_logprob(w) + inverse_log_det


# -----------------------------------------------------------------------------
# Conditioning encoder removed for non-amortized (single observation) training


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
    sigma_min: float = 0.01,
    sigma_max: float = 1.0,
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
    log_sigma_ratio = math.log(float(sigma_max) / float(sigma_min))

    t = 0.0
    for i in range(num_steps):
        # Continuous time
        t_curr = torch.full((B, 1), t, device=device, dtype=z0.dtype)
        # Compute sigma(t)
        sigma_t = default_sigma_schedule(t_curr, sigma_min, sigma_max)
        # Score: [B, d]
        with torch.enable_grad():
            s_t = score_fn(z_t, t_curr.reshape(B))
        # Drift for PF‑ODE
        g2_t = 2.0 * log_sigma_ratio * sigma_t.pow(2)
        drift = -0.5 * g2_t * s_t
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
    """Train the RealNVP posterior sampler for one fixed problem.

    This function orchestrates loading the pre‑trained VAE + SDF networks and
    score model, constructing the dataset with conditioning grids, building
    the RealNVP flow, and optimising the variational posterior using the
    PF‑ODE log prior.
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

    # The score model may have been trained with conditioning (cond_dim=8).
    # To load such a checkpoint robustly, instantiate with cond_dim=8 and
    # feed zeros for cond during prior evaluation.
    # Instantiate with cond_dim=0 to match an unconditional checkpoint (input dim = latent_dim + time_emb_dim)
    score_cond_dim = int(args.score_cond_dim) if int(args.score_cond_dim) > 0 else 0
    score_model = MLPDiffusionModel(latent_dim=args.latent_dim, hidden=args.hidden_dim, time_emb_dim=args.time_emb_dim, cond_dim=score_cond_dim).to(device)
    if os.path.exists(args.diffusion_ckpt):
        diff_ckpt = torch.load(args.diffusion_ckpt, map_location=device)
        try:
            if isinstance(diff_ckpt, dict) and "model_state_dict" in diff_ckpt:
                score_model.load_state_dict(diff_ckpt["model_state_dict"])
            elif isinstance(diff_ckpt, dict) and "ema_state_dict" in diff_ckpt:
                score_model.load_state_dict(diff_ckpt["ema_state_dict"])
            else:
                score_model.load_state_dict(diff_ckpt)
            print(f"Loaded diffusion model from {args.diffusion_ckpt}")
        except Exception as e:
            print(f"Failed to load diffusion checkpoint: {e}")
    else:
        print(f"Warning: diffusion checkpoint {args.diffusion_ckpt} not found. Using random init.")
    # Freeze the score model during posterior training
    score_model.eval()
    for p in score_model.parameters():
        p.requires_grad = False

    def score_fn(z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the score s_θ(z_t, t) required for the PF‑ODE.

        Here `t` is a 1‑D tensor of size [B] representing the continuous time in
        [0,1]. trainer_diffusion.py passes log sigma into the time embedding, so
        this wrapper computes log sigma(t) before calling the model. We call the
        model with `cond=None` because the prior is unconditional. The model was
        trained using a target of
            s_θ(z_t,t) ≈ -ε / σ(t)
        thus the output directly approximates the score.
        """
        # trainer_diffusion uses a VE-SDE with continuous time embedding via log sigma.
        # Compute log sigma(t) and pass as the time embedding; unconditional prior.
        t = t.reshape(-1).to(z_t.device)
        sigma_t = default_sigma_schedule(t, args.sigma_min, args.sigma_max)
        t_embed = torch.log(sigma_t)
        # Unconditional prior: no conditioning vector expected by the score model
        return score_model(z_t, t_embed, cond=None)

    # -------------------------------------------------------------------------
    # Single-observation setup: fix problem, forces, and design mask
    # -------------------------------------------------------------------------
    selto = SELTODataset(root=".", name=args.dataset_name, train=True)
    voxel_grids = create_voxel_grids(selto)
    problem_default, _ = selto[0]
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
    # Get first sample to obtain cond_grid, forces, and Ω_design; fix them
    sample = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)))
    if len(sample) != 6:
        raise RuntimeError("collate_fn must return 6 elements: pc, qp, sdf_vals, cond, F, Ω_design")
    _, _, _, cond_grid_1, forces_1, Omega_design_1 = sample
    if isinstance(cond_grid_1, list):
        cond_grid_1 = torch.stack(cond_grid_1, dim=0)
    # cond_grid_1: [1, C, D, H, W]; forces_1: list/tensor per sample; Omega_design_1: list/tensor per sample
    cond_grid_fixed = cond_grid_1.to(device)
    forces_fixed = forces_1.to(device)
    Omega_design_fixed = Omega_design_1.to(device)

    # -------------------------------------------------------------------------
    # Build unconditional RealNVP flow
    # -------------------------------------------------------------------------
    flow = RealNVP(latent_dim=args.latent_dim, hidden_dim=args.flow_hidden_dim, num_layers=args.flow_num_layers).to(device)
    optimizer = optim.Adam(list(flow.parameters()), lr=args.lr)

    # Pre-instantiate and preassemble a single FDM solver to reuse its tensors
    fdm_solver = FDM(padding_depth=0)
    try:
        fdm_solver.assemble_tensors(problem_default)
    except Exception:
        pass
    energy_baseline = None

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        flow.train()
        epoch_loss = 0.0
        num_batches = 0
        pbar = tqdm(range(args.iters_per_epoch), desc=f"Epoch {epoch}", leave=False)
        for _ in pbar:
            B = args.batch_size
            # Sample base noise and transform to latent z (unconditional flow)
            w = torch.randn(B, args.latent_dim, device=device)
            z, log_det = flow(w)

            # Compute energy functional on fixed single-observation parameters
            # Repeat fixed tensors to match batch size
            def repeat_batch(t: torch.Tensor, batch_size: int) -> torch.Tensor:
                reps = (batch_size,) + (1,) * (t.dim() - 1)
                return t.repeat(reps)

            cond_grid_rep = repeat_batch(cond_grid_fixed, B)
            forces_rep = repeat_batch(forces_fixed, B)
            Omega_design_rep = repeat_batch(Omega_design_fixed, B)
            # Compute energy with optional verbose timing
            if args.verbose:
                import time
                t0 = time.perf_counter()
            energy = energy_functional(
                z,
                cond_grid_rep,
                modulation_module=modulation_module,
                device=device,
                eval_resolution=tuple(args.eval_resolution),
                problem_default=problem_default,
                forces=forces_rep,
                Ω_design=Omega_design_rep,
                pde_solver=fdm_solver,
            )
            if args.verbose:
                t1 = time.perf_counter()
                print(f"Energy/PDE time: {(t1 - t0)*1000:.2f} ms for B={B}")

            # Compute log prior via PF‑ODE
            if args.verbose:
                import time
                t2 = time.perf_counter()
            logp = compute_log_prior_pf(
                z,
                score_fn,
                num_steps=args.pf_steps,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
            )
            if args.verbose:
                t3 = time.perf_counter()
                print(f"PF-ODE prior time: {(t3 - t2)*1000:.2f} ms for B={B}, steps={args.pf_steps}")
            # Flow log prob for reparameterized entropy/KL terms.
            logq = standard_normal_logprob(w) - log_det

            # The dl4to PDE energy is non-differentiable with respect to z here:
            # SDF evaluation is chunked through NumPy, thresholded to binary density,
            # and solved by a CPU FDM path. Use a score-function term so lower-energy
            # samples still update q_phi.
            energy_detached = energy.detach()
            batch_energy = float(energy_detached.mean().item())
            if energy_baseline is None:
                energy_baseline = batch_energy
            else:
                decay = float(args.energy_baseline_decay)
                energy_baseline = decay * energy_baseline + (1.0 - decay) * batch_energy
            energy_advantage = energy_detached - float(energy_baseline)
            logq_score = flow_log_prob(flow, z.detach())
            energy_score_loss = energy_advantage * logq_score

            # Loss gradient: score-function energy term + pathwise prior/KL terms.
            loss = energy_score_loss - logp + logq
            loss_mean = loss.mean()
            objective_mean = (energy_detached - logp.detach() + logq.detach()).mean()
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()
            epoch_loss += objective_mean.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss_mean.item():.4f}", "obj": f"{objective_mean.item():.4f}"})
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch}/{args.epochs} | Avg Loss: {avg_loss:.6f}")
        # Save checkpoint periodically
        if epoch % args.save_every == 0 or epoch == args.epochs:
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt_path = os.path.join(args.out_dir, f"posterior_epoch_{epoch}.pth")
            torch.save(
                {
                    "flow_state_dict": flow.state_dict(),
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
                        help="Kept for checkpoint metadata compatibility; PF-ODE uses --pf-steps")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs for the posterior")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for posterior training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for the RealNVP flow")
    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="Hidden dimension of the score network (unused here)")
    parser.add_argument("--time-emb-dim", type=int, default=128,
                        help="Dimension of time embedding for score network")
    # Remove conditioning embedding since we train per-observation (non-amortized)
    parser.add_argument("--flow-hidden-dim", type=int, default=512,
                        help="Hidden dimension in RealNVP coupling nets")
    parser.add_argument("--flow-num-layers", type=int, default=6,
                        help="Number of affine coupling layers in RealNVP")
    parser.add_argument("--sigma-min", type=float, default=0.01,
                        help="Minimum sigma for VE SDE (PF‑ODE)")
    parser.add_argument("--sigma-max", type=float, default=1.0,
                        help="Maximum sigma for VE SDE (PF‑ODE)")
    parser.add_argument("--pf-steps", type=int, default=50,
                        help="Number of Euler steps for PF‑ODE integration")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--out-dir", type=str, default="checkpoints_posterior",
                        help="Output directory for posterior checkpoints")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    # Additional control: iterations per epoch for unconditional training loop
    parser.add_argument("--iters-per-epoch", type=int, default=200,
                        help="Number of optimization steps per epoch without a dataloader")
    # Evaluation resolution for PDE grid (D H W)
    parser.add_argument("--eval-resolution", nargs=3, type=int, default=[39, 39, 21],
                        help="Per-axis resolution (D H W) for SDF-to-density evaluation")
    # Verbose timing for energy/PDE and PF-ODE prior
    parser.add_argument("--verbose", action="store_true", help="Print per-iteration timing diagnostics")
    parser.add_argument("--score-cond-dim", type=int, default=0,
                        help="Set to 8 when loading a score checkpoint trained with trainer_diffusion.py --cond")
    parser.add_argument("--energy-baseline-decay", type=float, default=0.9,
                        help="EMA decay for the score-function energy baseline")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
