import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from models import *
from utils.preprocess_data import *
import dl4to
from dl4to.datasets import SELTODataset
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import math
import os
import argparse
import csv
import json
torch.cuda.empty_cache()  # After each step

def sdf_loss_function(sdf_pred, sdf_gt):
    """
    Loss matching the paper's description:
    - L1 loss between predicted and ground-truth signed distances.

    Returns:
        total_loss (tensor), dict with components
    """
    # Ensure shapes: [B, N] or [B, N, 1] -> squeeze last dim
    # Backwards-compatible signature: accept optional weights later via wrapper
    sdf_pred_clamped = sdf_pred
    sdf_gt_clamped = sdf_gt

    # L1 loss per the paper
    loss_l1 = F.l1_loss(sdf_pred_clamped.squeeze(), sdf_gt_clamped.squeeze(), reduction='mean')

    return loss_l1, {"l1": loss_l1.item(), "mean_weight": 1.0}


def compute_sdf_focus_weights(sdf_gt, mode='gauss', alpha=5.0, sigma=0.03, tau=0.05, eps=1e-6, max_weight=50.0, normalize=True):
    """
    Compute per-point importance weights biased toward |sdf_gt| ~ 0 (surface).

    Args:
        sdf_gt: Tensor [B, N] (normalized SDF)
        mode: 'gauss' (recommended), 'linear', or 'inv'
        alpha: strength multiplier for the bump
        sigma: width for gaussian (used when mode='gauss')
        tau: width for linear ramp (used when mode='linear')
        max_weight: clamp to avoid extreme weights
        normalize: if True, normalize per-sample mean weight to 1.0

    Returns:
        weights: Tensor [B, N]
    """
    sdf_abs = sdf_gt.abs()
    if mode == 'gauss':
        w = 1.0 + alpha * torch.exp(- (sdf_abs ** 2) / (2.0 * (sigma ** 2)))
    elif mode == 'linear':
        w = 1.0 + alpha * torch.clamp(1.0 - sdf_abs / (tau + eps), min=0.0)
    elif mode == 'inv':
        w = 1.0 + alpha / (sdf_abs + eps)
    else:
        raise ValueError(f"unknown focus mode {mode}")

    if max_weight is not None:
        w = torch.clamp(w, max=max_weight)

    if normalize:
        mean_w = w.mean(dim=1, keepdim=True)
        w = w / (mean_w + 1e-8)

    return w


def sdf_loss_weighted(sdf_pred, sdf_gt, weights=None):
    """Weighted L1 SDF loss. If weights is None, fallback to standard mean L1.

    Returns (loss_tensor, dict)
    """
    sdf_pred = sdf_pred.squeeze()
    sdf_gt = sdf_gt.squeeze()
    if weights is None:
        loss_l1 = F.l1_loss(sdf_pred, sdf_gt, reduction='mean')
        mean_weight = 1.0
    else:
        err = (sdf_pred - sdf_gt).abs()
        weights = weights.to(err.device, dtype=err.dtype)
        weighted_sum = (weights * err).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1e-8)
        per_sample = weighted_sum / denom
        loss_l1 = per_sample.mean()
        mean_weight = float(weights.mean().item())

    return loss_l1, {"l1": loss_l1.item(), "mean_weight": mean_weight}

def compute_sdf(query_points, surface_points, normals, epsilon=1e-8):
    """
    Compute SDF with guaranteed [B, N, 1] output shape
    Args:
        query_points: [B, N, 3]
        surface_points: [B, M, 3] 
        normals: [B, M, 3]
    Returns:
        sdf: [B, N, 1]
    """
    B, N, _ = query_points.shape
    M = surface_points.shape[1]
    
    # 1. Pairwise differences [B, N, M, 3]
    diffs = query_points.unsqueeze(2) - surface_points.unsqueeze(1)
    
    # 2. Distances [B, N, M]
    distances = torch.norm(diffs, dim=-1)
    del diffs
    # 3. Find nearest points [B, N]
    min_dist, min_idx = torch.min(distances, dim=2)
    min_idx = min_idx.unsqueeze(-1).expand(-1, -1, 3)  # [B, N, 3]
    
    # 4. Gather nearest points and normals
    nearest_points = torch.gather(surface_points, 1, min_idx)
    nearest_normals = torch.gather(normals, 1, min_idx)
    
    # 5. Compute sign
    vectors = query_points - nearest_points  # [B, N, 3]
    dot = torch.sum(vectors * nearest_normals, dim=-1)  # [B, N]
    sign = torch.sign(dot)
    del nearest_points
    del nearest_normals
    # 6. Final SDF [B, N]
    return (min_dist * sign).squeeze()


def _write_loss_svg(history, filename):
    metrics = [
        ("sdf", "SDF", "#1f77b4"),
        ("kl_weighted", "beta*KL", "#d62728"),
        ("recon", "Recon", "#2ca02c"),
        ("total", "Total", "#111111"),
    ]
    width, height = 760, 430
    left, right, top, bottom = 64, 24, 42, 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    epochs = [row["epoch"] for row in history]
    values = [
        float(row[key])
        for key, _, _ in metrics
        for row in history
        if row.get(key) is not None
    ]
    y_min = 0.0
    y_max = max(values) if values else 1.0
    if y_max <= y_min:
        y_max = 1.0
    y_max *= 1.05

    def xy(index, value):
        x = left + (plot_w * index / max(1, len(history) - 1))
        y = top + plot_h - ((float(value) - y_min) / (y_max - y_min) * plot_h)
        return x, y

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" viewBox="0 0 {0} {1}">'.format(width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="{0}" y="24" font-family="sans-serif" font-size="18" font-weight="700">Training Loss History</text>'.format(left),
        '<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="#333"/>'.format(left, top + plot_h, left + plot_w),
        '<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" stroke="#333"/>'.format(left, top, top + plot_h),
    ]

    for tick in range(6):
        frac = tick / 5.0
        y = top + plot_h - frac * plot_h
        value = y_min + frac * (y_max - y_min)
        lines.append('<line x1="{0}" y1="{1:.2f}" x2="{2}" y2="{1:.2f}" stroke="#e8e8e8"/>'.format(left, y, left + plot_w))
        lines.append('<text x="{0}" y="{1:.2f}" font-family="sans-serif" font-size="11" text-anchor="end">{2:.3g}</text>'.format(left - 8, y + 4, value))

    if epochs:
        for index in sorted(set([0, len(epochs) - 1])):
            x, _ = xy(index, 0.0)
            lines.append('<text x="{0:.2f}" y="{1}" font-family="sans-serif" font-size="11" text-anchor="middle">{2}</text>'.format(x, height - 26, epochs[index]))
        lines.append('<text x="{0}" y="{1}" font-family="sans-serif" font-size="12" text-anchor="middle">epoch</text>'.format(left + plot_w / 2, height - 8))

    legend_x = left
    for key, label, color in metrics:
        lines.append('<rect x="{0}" y="{1}" width="12" height="12" fill="{2}"/>'.format(legend_x, height - 48, color))
        lines.append('<text x="{0}" y="{1}" font-family="sans-serif" font-size="12">{2}</text>'.format(legend_x + 18, height - 38, label))
        legend_x += 116
        points = [xy(index, row[key]) for index, row in enumerate(history)]
        if points:
            point_text = " ".join("{0:.2f},{1:.2f}".format(x, y) for x, y in points)
            lines.append('<polyline points="{0}" fill="none" stroke="{1}" stroke-width="2"/>'.format(point_text, color))

    lines.append("</svg>")
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def write_loss_artifacts(history, artifact_dir):
    if not artifact_dir or not history:
        return
    os.makedirs(artifact_dir, exist_ok=True)
    fields = ["epoch", "sdf", "kl", "kl_weighted", "recon", "sign", "mean_weight", "total"]
    csv_path = os.path.join(artifact_dir, "loss_history.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field) for field in fields})

    json_path = os.path.join(artifact_dir, "loss_history.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    _write_loss_svg(history, os.path.join(artifact_dir, "loss_plot.svg"))


def write_run_config(args, artifact_dir, dataset_names, checkpoint_dir, vae_checkpoint_dir):
    if not artifact_dir:
        return
    os.makedirs(artifact_dir, exist_ok=True)
    config = {
        "run_name": args.run_name or safe_run_name(args.dataset_name),
        "dataset_name_arg": args.dataset_name,
        "resolved_datasets": dataset_names,
        "checkpoint_dir": checkpoint_dir,
        "vae_checkpoint_dir": vae_checkpoint_dir,
        "training_args": vars(args),
    }
    with open(os.path.join(artifact_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


# def staged_training(modulation_module, train_dataloader, device, num_epochs_stage_1, num_epochs_stage_2):
#     """
#     Full staged training pipeline.
#     """
def staged_training(modulation_module, train_dataloader, device, num_epochs =10000, beta_kl=1e-4, prior_std=0.25, lr=1e-4,
                    sdf_focus=False, sdf_focus_alpha=5.0, sdf_focus_sigma=0.03, sdf_focus_mode='gauss',
                    sdf_focus_max_weight=50.0, sdf_focus_normalize=True,
                    sdf_sign_loss=False, sdf_sign_gamma=1.0, sdf_sign_margin=0.01, sdf_sign_threshold=0.03,
                    checkpoint_dir="checkpoints_mod", vae_checkpoint_dir="checkpoints_vae", save_every=10,
                    artifact_dir=None):
    """
    Joint training loop that optimizes the SDF L1 loss and the VAE KL regularizer
    together. The function runs for `num_epochs` epochs.

    Per-batch loss: L = L_sdf + beta_kl * KL(q(z|x) || N(0, prior_std^2)).
    We do NOT include the VAE reconstruction loss in the default joint objective
    to match the paper's Stage-2 emphasis unless the network explicitly returns
    a recon and you want to include it (we still log it if available).
    """
    total_epochs = num_epochs

    optimizer = optim.Adam(modulation_module.parameters(), lr=lr)
    scaler = GradScaler()

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(vae_checkpoint_dir, exist_ok=True)
    history = []

    print(f"Joint training for {total_epochs} epochs (SDF + beta_kl*KL). beta_kl={beta_kl}, prior_std={prior_std}")

    for epoch in range(total_epochs):
        modulation_module.train()
        epoch_sdf_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_sign_loss = 0.0
        epoch_mean_weight = 0.0
        epoch_weight_batches = 0

        for batch_idx, batch in enumerate(train_dataloader):
            # Expect batches like (point_clouds, query_points, sdf_gt)
            if len(batch) == 3:
                point_clouds, query_points, sdf_gt = batch
            elif len(batch) == 2:
                point_clouds, query_points = batch
                sdf_gt = None
            else:
                # fallback: treat whole batch as point_clouds
                point_clouds = batch[0]
                query_points = None
                sdf_gt = None

            # stack point clouds and move tensors to device
            try:
                point_clouds = torch.stack(point_clouds).to(device)
            except Exception:
                # if point_clouds already tensor
                point_clouds = point_clouds.to(device)

            if query_points is not None:
                query_points = query_points.to(device)

            if sdf_gt is not None:
                sdf_gt = sdf_gt.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = modulation_module(point_clouds, query_points) if query_points is not None else modulation_module(point_clouds)

                # Attempt to extract commonly expected outputs. We support a few
                # possible output shapes/ordering but expect at minimum: sdf_pred, z, latent_pc, x_recon, mu, logvar
                sdf_pred = None
                mu = None
                logvar = None
                recon = None
                latent_pc = None

                if isinstance(outputs, (tuple, list)):
                    # find values by type/shape heuristics
                    # common pattern: (sdf_pred, z, latent_pc, x_recon, mu, logvar)
                    if len(outputs) >= 1:
                        sdf_pred = outputs[0]
                    if len(outputs) >= 3:
                        latent_pc = outputs[2]
                    if len(outputs) >= 5:
                        mu = outputs[-2]
                        logvar = outputs[-1]
                    # try to find recon if present
                    if len(outputs) >= 4:
                        recon = outputs[3]
                elif isinstance(outputs, dict):
                    sdf_pred = outputs.get('sdf_pred') or outputs.get('sdf')
                    mu = outputs.get('mu')
                    logvar = outputs.get('logvar')
                    recon = outputs.get('x_recon') or outputs.get('recon')
                else:
                    # single tensor output -> treat as sdf_pred
                    sdf_pred = outputs

                if sdf_pred is None:
                    raise RuntimeError("modulation_module did not return an SDF prediction as first/only output")

                if sdf_pred.dim() == 3:
                    sdf_pred = sdf_pred.squeeze(-1)

                # SDF loss
                if sdf_gt is None:
                    raise RuntimeError("Dataset must provide ground-truth SDFs for joint training")

                if sdf_focus and (sdf_gt is not None):
                    weights = compute_sdf_focus_weights(sdf_gt, mode=sdf_focus_mode,
                                                        alpha=sdf_focus_alpha, sigma=sdf_focus_sigma,
                                                        max_weight=sdf_focus_max_weight, normalize=sdf_focus_normalize)
                    sdf_loss, loss_dict = sdf_loss_weighted(sdf_pred, sdf_gt, weights=weights)
                    # accumulate mean weight for logging
                    epoch_mean_weight += float(loss_dict.get('mean_weight', 1.0))
                    epoch_weight_batches += 1
                else:
                    sdf_loss, loss_dict = sdf_loss_function(sdf_pred, sdf_gt)

                # KL loss (if mu/logvar available)
                if (mu is not None) and (logvar is not None):
                    sigma2 = logvar.exp()
                    prior_var = prior_std ** 2
                    kl_per_sample = 0.5 * ((sigma2 + mu.pow(2)) / prior_var - 1 - logvar + math.log(prior_var)).sum(dim=1)
                    kl = kl_per_sample.mean()
                else:
                    kl = torch.tensor(0.0, device=sdf_loss.device)

                # Optional: sign-consistency hinge loss (penalize sign flips near interface)
                sign_loss = torch.tensor(0.0, device=sdf_loss.device)
                # sign-loss is kept external and configurable via CLI (see main)
                if sdf_sign_loss:
                    # mask points near true surface
                    mask = (sdf_gt.abs() <= sdf_sign_threshold)
                    if mask.any():
                        prod = sdf_pred * sdf_gt
                        # hinge penalty when prod is negative (opposite sign) or below margin
                        hinge = F.relu(-prod + sdf_sign_margin)
                        # only consider masked points
                        hinge_masked = hinge[mask]
                        if hinge_masked.numel() > 0:
                            sign_loss = hinge_masked.mean()
                # Combined loss: SDF + beta * KL + optional sign term
                loss = sdf_loss + beta_kl * kl + sdf_sign_gamma * sign_loss

                # optionally collect recon loss for logging
                if recon is not None and 'latent_pc' in locals():
                    try:
                        recon_loss = F.mse_loss(recon, latent_pc, reduction='mean')
                    except Exception:
                        recon_loss = torch.tensor(0.0, device=loss.device)
                else:
                    recon_loss = torch.tensor(0.0, device=loss.device)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_sdf_loss += sdf_loss.item()
            epoch_kl_loss += (kl.item() if isinstance(kl, torch.Tensor) else float(kl))
            epoch_recon_loss += (recon_loss.item() if isinstance(recon_loss, torch.Tensor) else float(recon_loss))
            epoch_sign_loss += (sign_loss.item() if isinstance(sign_loss, torch.Tensor) else float(sign_loss))

        avg_sdf = epoch_sdf_loss / len(train_dataloader)
        avg_kl = epoch_kl_loss / len(train_dataloader)
        avg_recon = epoch_recon_loss / len(train_dataloader)
        avg_sign = epoch_sign_loss / len(train_dataloader)
        avg_mean_weight = epoch_mean_weight / epoch_weight_batches if epoch_weight_batches > 0 else 1.0
        history.append(
            {
                "epoch": epoch + 1,
                "sdf": float(avg_sdf),
                "kl": float(avg_kl),
                "kl_weighted": float(beta_kl * avg_kl),
                "recon": float(avg_recon),
                "sign": float(avg_sign),
                "mean_weight": float(avg_mean_weight),
                "total": float(avg_sdf + beta_kl * avg_kl + sdf_sign_gamma * avg_sign),
            }
        )

        if epoch_weight_batches > 0:
            print(f"Epoch [{epoch + 1}/{total_epochs}] - SDF: {avg_sdf:.6f}, KL: {avg_kl:.6f}, Recon(logged): {avg_recon:.6f}, sign_loss: {avg_sign:.6f}, mean_weight: {avg_mean_weight:.4f}")
        else:
            print(f"Epoch [{epoch + 1}/{total_epochs}] - SDF: {avg_sdf:.6f}, KL: {avg_kl:.6f}, Recon(logged): {avg_recon:.6f}, sign_loss: {avg_sign:.6f}")

        # periodic checkpointing
        if (epoch + 1) % save_every == 0 or (epoch + 1) == total_epochs:
            save_checkpoint(modulation_module, optimizer, epoch, {'sdf': avg_sdf, 'kl': avg_kl}, os.path.join(checkpoint_dir, "mod_last.pth"))
            save_checkpoint(modulation_module.vae, optimizer, epoch, {'recon': avg_recon, 'kl': avg_kl}, os.path.join(vae_checkpoint_dir, "vae_last.pth"))

        write_loss_artifacts(history, artifact_dir)
        torch.cuda.empty_cache()

    return history

def train_diffusion_model(diffusion_model, modulation_module, dataloader, optimizer, device, timesteps, betas, alphas_cumprod, num_epochs):
    modulation_module.eval()  # Freeze the modulation module
    diffusion_model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (point_clouds, query_points, normals) in enumerate(dataloader):
            point_clouds = [pc.to(device) for pc in point_clouds]
            point_clouds = torch.stack(point_clouds)
            query_points = query_points.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through the modulation module (no gradients)
            with torch.no_grad():
                outputs = modulation_module(point_clouds, query_points)
                sdf_values, z, latent_pc, x_recon = outputs[:4]
            
            # Sample a random time step `t`
            t = torch.randint(0, timesteps, (z.size(0),), device=device)
            
            # Forward process: add noise to `z`
            z_t, noise = forward_process(z, t, betas, alphas_cumprod)
            
            # Forward pass through the diffusion model
            predicted_noise = diffusion_model(z_t, t)
            
            # Compute diffusion loss (MSE between predicted and actual noise)
            diffusion_loss = F.mse_loss(predicted_noise, noise)
            
            # Backward pass and optimization
            diffusion_loss.backward()
            optimizer.step()
            
            # Accumulate loss
            total_loss += diffusion_loss.item()
        
        # Print epoch loss
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Diffusion Model Loss: {avg_loss:.4f}")

def save_checkpoint(model, optimizer, epoch, loss, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename, device):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
    return start_epoch, checkpoint.get('loss', None)


def resolve_selto_dataset_names(dataset_name):
    names = [name.strip() for name in dataset_name.split(",") if name.strip()]
    if len(names) == 1 and names[0].lower() in {"all", "combined", "combined_all"}:
        return ["disc_simple", "disc_complex", "sphere_simple", "sphere_complex"]
    return names


def safe_run_name(dataset_name):
    names = resolve_selto_dataset_names(dataset_name)
    if len(names) == 4 and set(names) == {"disc_simple", "disc_complex", "sphere_simple", "sphere_complex"}:
        return "combined_all"
    return "_plus_".join(names)


def load_selto_voxel_grids(dataset_name, max_samples_per_dataset=0):
    voxel_grids = []
    names = resolve_selto_dataset_names(dataset_name)
    for name in names:
        print(f"Loading SELTO dataset '{name}'...")
        selto_size = max_samples_per_dataset if max_samples_per_dataset and max_samples_per_dataset > 0 else -1
        selto = SELTODataset(root='.', name=name, train=True, size=selto_size)
        print(f"SELTO dataset '{name}' loaded with {len(selto)} samples.")
        print(f"Constructing voxel grids for '{name}'...")
        voxel_grids.extend(create_voxel_grids(selto))
        if max_samples_per_dataset and max_samples_per_dataset > 0:
            print(f"Voxel grids for '{name}' constructed from {min(max_samples_per_dataset, len(selto))} samples.")
        else:
            print(f"Voxel grids for '{name}' constructed.")
        torch.cuda.empty_cache()
    return names, voxel_grids

def main():
    # Single argparse parser for all hyperparameters (defaults preserved)
    parser = argparse.ArgumentParser(description='Train modulation module (VAE + SDF) with optional SDF focus and sign losses')
    parser.add_argument('--encoding-dim', type=int, default=256, help='Feature size produced by encoder and used by decoder conditioning vector')
    parser.add_argument('--latent-dim', type=int, default=64, help='Compressed latent z dimension for the VAE')
    parser.add_argument('--num-epochs', type=int, default=10000, help='Total training epochs for joint AE+SDF')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Optimizer learning rate')
    parser.add_argument('--beta-kl', type=float, default=1e-5, help='KL regularization weight')
    parser.add_argument('--prior-std', type=float, default=0.25, help='Standard deviation of the Gaussian VAE prior')
    parser.add_argument('--timesteps', type=int, default=1000, help='Kept for compatibility; modulation training does not use this')
    parser.add_argument('--diffusion-steps', type=int, default=100, help='Kept for compatibility; modulation training does not use this')
    # SDF focus weighting
    parser.add_argument('--sdf_focus', action='store_true', help='Enable focus weighting toward SDF zero-crossing')
    parser.add_argument('--sdf_focus_alpha', type=float, default=5.0, help='Alpha multiplier for focus bump')
    parser.add_argument('--sdf_focus_sigma', type=float, default=0.03, help='Gaussian sigma (normalized SDF units)')
    parser.add_argument('--sdf_focus_mode', type=str, default='gauss', choices=['gauss', 'linear', 'inv'], help='Weighting mode')
    parser.add_argument('--sdf_focus_max_weight', type=float, default=50.0, help='Max clamp for per-point weight')
    parser.add_argument('--no_sdf_focus_normalize', dest='sdf_focus_normalize', action='store_false', help='Disable per-sample normalization of weights')
    parser.set_defaults(sdf_focus_normalize=True)
    # Sign-consistency hinge loss
    parser.add_argument('--sdf_sign_loss', action='store_true', help='Enable hinge-style sign consistency loss near SDF=0')
    parser.add_argument('--sdf_sign_gamma', type=float, default=100.0, help='Weight for sign-consistency loss term')
    parser.add_argument('--sdf_sign_margin', type=float, default=0.01, help='Margin used in hinge for sign loss')
    parser.add_argument('--sdf_sign_threshold', type=float, default=0.1, help='Consider points with |sdf_gt| <= threshold for sign loss')
    parser.add_argument('--dataset-name', type=str, default='sphere_complex',
                        help='SELTO dataset name, comma-separated names, or all/combined_all')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name used for output checkpoint subdirectories; defaults from --dataset-name')
    parser.add_argument('--checkpoint-root', type=str, default='checkpoints_mod')
    parser.add_argument('--vae-checkpoint-root', type=str, default='checkpoints_vae')
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-samples-per-dataset', type=int, default=0,
                        help='Maximum SELTO training samples to use from each dataset; 0 uses the full split')
    parser.add_argument('--num-query-points', type=int, default=5000)
    parser.add_argument('--fixed-surface-points-size', type=int, default=10000)
    parser.add_argument('--noise-std', type=float, default=0.1)
    parser.add_argument('--artifact-dir', type=str, default=None,
                        help='Directory for run_config.json, loss_history.csv/json, and loss_plot.svg')
    args, unknown = parser.parse_known_args()

    # Assign from args (keeping default values if not provided)
    encoding_dim = int(args.encoding_dim)
    latent_dim = int(args.latent_dim)
    num_epochs = int(args.num_epochs)
    learning_rate = float(args.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()  # After each step
    dataset_names, voxel_grids = load_selto_voxel_grids(args.dataset_name, max_samples_per_dataset=args.max_samples_per_dataset)
    run_name = args.run_name or safe_run_name(args.dataset_name)
    checkpoint_dir = os.path.join(args.checkpoint_root, run_name)
    vae_checkpoint_dir = os.path.join(args.vae_checkpoint_root, run_name)
    print(f"Training run '{run_name}' on datasets: {', '.join(dataset_names)}")
    print(f"Modulation checkpoints: {checkpoint_dir}")
    print(f"VAE checkpoints: {vae_checkpoint_dir}")
    if args.artifact_dir:
        print(f"Artifacts: {args.artifact_dir}")
        write_run_config(args, args.artifact_dir, dataset_names, checkpoint_dir, vae_checkpoint_dir)
    torch.cuda.empty_cache()  # After each step
    # Create the dataset and DataLoader
    dataset = VoxelSDFDataset(voxel_grids, num_query_points=args.num_query_points, fixed_surface_points_size=args.fixed_surface_points_size, noise_std=args.noise_std, device=device)

    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    vae = ImprovedVAE(input_dim=encoding_dim, latent_dim=latent_dim, hidden_dim=512, num_layers=8).to(device)
    sdf_network = ImprovedSDFNetwork(input_dim=encoding_dim, latent_dim=encoding_dim, hidden_dim=128, output_dim=1, num_layers=8).to(device)
    modulation_module = ModulationModule(vae, sdf_network).to(device)
    staged_training(modulation_module, train_dataloader, device, num_epochs=num_epochs, beta_kl=args.beta_kl,
                    prior_std=args.prior_std,
                    lr=learning_rate,
                    sdf_focus=args.sdf_focus,
                    sdf_focus_alpha=args.sdf_focus_alpha,
                    sdf_focus_sigma=args.sdf_focus_sigma,
                    sdf_focus_mode=args.sdf_focus_mode,
                    sdf_focus_max_weight=args.sdf_focus_max_weight,
                    sdf_focus_normalize=args.sdf_focus_normalize,
                    sdf_sign_loss=args.sdf_sign_loss,
                    sdf_sign_gamma=args.sdf_sign_gamma,
                    sdf_sign_margin=args.sdf_sign_margin,
                    sdf_sign_threshold=args.sdf_sign_threshold,
                    checkpoint_dir=checkpoint_dir,
                    vae_checkpoint_dir=vae_checkpoint_dir,
                    save_every=args.save_every,
                    artifact_dir=args.artifact_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(modulation_module.state_dict(), os.path.join(checkpoint_dir, "modulation_module.pth"))

if __name__ == "__main__":
    torch.cuda.empty_cache()  # After each step
    main()
