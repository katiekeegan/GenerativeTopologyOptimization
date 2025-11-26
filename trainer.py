import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from models import *
from utils.preprocess_data import *
import dl4to  # now this import won't fail
from dl4to.datasets import SELTODataset
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import math
import os
import argparse
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

# def staged_training(modulation_module, train_dataloader, device, num_epochs_stage_1, num_epochs_stage_2):
#     """
#     Full staged training pipeline.
#     """
def staged_training(modulation_module, train_dataloader, device, num_epochs_stage_1, num_epochs_stage_2, beta_kl=1e-4, prior_std=0.25, lr=1e-4,
                    sdf_focus=False, sdf_focus_alpha=5.0, sdf_focus_sigma=0.03, sdf_focus_mode='gauss',
                    sdf_focus_max_weight=50.0, sdf_focus_normalize=True,
                    sdf_sign_loss=False, sdf_sign_gamma=1.0, sdf_sign_margin=0.01, sdf_sign_threshold=0.03):
    """
    Joint training loop that optimizes the SDF L1 loss and the VAE KL regularizer
    together. The function runs for `num_epochs_stage_1 + num_epochs_stage_2` epochs
    but performs the same joint update every epoch (so effectively it's a single
    training loop that lasts `total_epochs`).

    Per-batch loss: L = L_sdf + beta_kl * KL(q(z|x) || N(0, prior_std^2)).
    We do NOT include the VAE reconstruction loss in the default joint objective
    to match the paper's Stage-2 emphasis unless the network explicitly returns
    a recon and you want to include it (we still log it if available).
    """
    total_epochs = int(num_epochs_stage_1) + int(num_epochs_stage_2)

    optimizer = optim.Adam(modulation_module.parameters(), lr=lr)
    scaler = GradScaler()

    os.makedirs("checkpoints_mod", exist_ok=True)
    os.makedirs("checkpoints_vae", exist_ok=True)

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
                # Combined loss: SDF + beta * KL + sign term
                loss = sdf_loss + sdf_sign_gamma * sign_loss

                # optionally collect recon loss for logging
                if recon is not None and 'latent_pc' in locals():
                    try:
                        recon_loss = F.mse_loss(recon, latent_pc, reduction='mean')
                    except Exception:
                        recon_loss = torch.tensor(0.0, device=loss.device)
                else:
                    recon_loss = torch.tensor(0.0, device=loss.device)
                    loss = loss + 0.1*recon_loss

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

        if epoch_weight_batches > 0:
            avg_mean_weight = epoch_mean_weight / epoch_weight_batches
            print(f"Epoch [{epoch + 1}/{total_epochs}] - SDF: {avg_sdf:.6f}, KL: {avg_kl:.6f}, Recon(logged): {avg_recon:.6f}, sign_loss: {epoch_sign_loss/len(train_dataloader):.6f}, mean_weight: {avg_mean_weight:.4f}")
        else:
            print(f"Epoch [{epoch + 1}/{total_epochs}] - SDF: {avg_sdf:.6f}, KL: {avg_kl:.6f}, Recon(logged): {avg_recon:.6f}, sign_loss: {epoch_sign_loss/len(train_dataloader):.6f}")

        # periodic checkpointing
        if (epoch + 1) % 10 == 0:
            save_checkpoint(modulation_module, optimizer, epoch, {'sdf': avg_sdf, 'kl': avg_kl}, os.path.join("checkpoints_mod", "mod_last.pth"))
            save_checkpoint(modulation_module.vae, optimizer, epoch, {'recon': avg_recon, 'kl': avg_kl}, os.path.join("checkpoints_vae", "vae_last.pth"))

        torch.cuda.empty_cache()

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

def main():
    # train_modulation_module = True
    # Training hyperparameters
    encoding_dim = 256
    latent_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10000
    learning_rate = 1e-4
    diffusion_steps = 100
    # Example usage
    timesteps = 1000
    betas = cosine_beta_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
    save_path = 'trained_diffusion_sdf_model.pth'
    torch.cuda.empty_cache()  # After each step
    print("Loading SELTO dataset...")
    selto = SELTODataset(root='.', name='sphere_complex', train=True)
    print("SELTO dataset loaded!")
    print("Constructing voxel grids...")
    voxel_grids = create_voxel_grids(selto)
    print("Voxel grids constructed!")
    torch.cuda.empty_cache()  # After each step
    # Create the dataset and DataLoader
    dataset = VoxelSDFDataset(voxel_grids, num_query_points=5000,fixed_surface_points_size=10000, noise_std=0.1, device=device)

    # Quick diagnostics: compute mins/maxes/means across the stored SDF grids
    try:
        print("Computing SDF grid statistics from VoxelSDFDataset.sdf_grids...")
        all_mins = [float(g.min().item()) for g in dataset.sdf_grids]
        all_maxs = [float(g.max().item()) for g in dataset.sdf_grids]
        all_means = [float(g.mean().item()) for g in dataset.sdf_grids]
        global_min = float(np.min(all_mins)) if len(all_mins) > 0 else float('nan')
        global_max = float(np.max(all_maxs)) if len(all_maxs) > 0 else float('nan')
        mean_of_means = float(np.mean(all_means)) if len(all_means) > 0 else float('nan')
        print(f"SDF grids: count={len(all_mins)}, global_min={global_min:.6f}, global_max={global_max:.6f}, mean_of_means={mean_of_means:.6f}")

        # Also show normalized stats by dataset.sdf_scale if available
        sdf_scale = getattr(dataset, 'sdf_scale', None)
        if sdf_scale is not None and sdf_scale != 0:
            norm_mins = [m / float(sdf_scale) for m in all_mins]
            norm_maxs = [M / float(sdf_scale) for M in all_maxs]
            print(f"Normalized by sdf_scale={sdf_scale:.6f}: norm_global_min={float(np.min(norm_mins)):.6f}, norm_global_max={float(np.max(norm_maxs)):.6f}")
    except Exception as e:
        print(f"[trainer] Warning: failed to compute SDF grid statistics: {e}")

    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    # modulation_module = ModulationModule(pointnet_input_dim=3, pointnet_output_dim=256, latent_dim=128).to(device)
    # Correct shapes: VAE input_dim == feature size produced by PointNet (encoding_dim)
    # and VAE latent_dim == compressed z-size (latent_dim). The SDF network is
    # conditioned on the VAE decoder output (x_recon) which has size == encoding_dim,
    # therefore sdf_network.latent_dim should be encoding_dim (conditioning vector size).
    vae = ImprovedVAE(input_dim=encoding_dim, latent_dim=latent_dim, hidden_dim=512, num_layers=8).to(device)
    sdf_network = ImprovedSDFNetwork(input_dim=encoding_dim, latent_dim=encoding_dim, hidden_dim=128, output_dim=1, num_layers=8).to(device)
    modulation_module = ModulationModule(vae, sdf_network).to(device)
    optimizer = torch.optim.Adam(modulation_module.parameters(), lr=1e-4)
    # parse optional CLI args for SDF focus weighting (defaults keep original behavior)
    parser = argparse.ArgumentParser(description='Train modulation module with optional SDF focus weighting')
    parser.add_argument('--sdf_focus', action='store_true', help='Enable focus weighting toward SDF zero-crossing')
    parser.add_argument('--sdf_focus_alpha', type=float, default=5.0, help='Alpha multiplier for focus bump')
    parser.add_argument('--sdf_focus_sigma', type=float, default=0.03, help='Gaussian sigma (normalized SDF units)')
    parser.add_argument('--sdf_focus_mode', type=str, default='gauss', choices=['gauss', 'linear', 'inv'], help='Weighting mode')
    parser.add_argument('--sdf_focus_max_weight', type=float, default=50.0, help='Max clamp for per-point weight')
    parser.add_argument('--no_sdf_focus_normalize', dest='sdf_focus_normalize', action='store_false', help='Disable per-sample normalization of weights')
    parser.set_defaults(sdf_focus_normalize=True)
    # optional sign-consistency hinge loss to encourage correct sign near interface
    parser.add_argument('--sdf_sign_loss', action='store_true', help='Enable hinge-style sign consistency loss near SDF=0')
    parser.add_argument('--sdf_sign_gamma', type=float, default=1.0, help='Weight for sign-consistency loss term')
    parser.add_argument('--sdf_sign_margin', type=float, default=0.01, help='Margin used in hinge for sign loss')
    parser.add_argument('--sdf_sign_threshold', type=float, default=0.1, help='Consider points with |sdf_gt| <= threshold for sign loss')
    args, unknown = parser.parse_known_args()

    staged_training(modulation_module, train_dataloader, device, num_epochs= num_epochs, beta_kl=1e-5,
                    sdf_focus=args.sdf_focus,
                    sdf_focus_alpha=args.sdf_focus_alpha,
                    sdf_focus_sigma=args.sdf_focus_sigma,
                    sdf_focus_mode=args.sdf_focus_mode,
                    sdf_focus_max_weight=args.sdf_focus_max_weight,
                    sdf_focus_normalize=args.sdf_focus_normalize,
                    sdf_sign_loss=args.sdf_sign_loss,
                    sdf_sign_gamma=args.sdf_sign_gamma,
                    sdf_sign_margin=args.sdf_sign_margin,
                    sdf_sign_threshold=args.sdf_sign_threshold)
    torch.save(modulation_module.state_dict(), "modulation_module.pth")

if __name__ == "__main__":
    torch.cuda.empty_cache()  # After each step
    main()