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
torch.cuda.empty_cache()  # After each step

def sdf_loss_function(sdf_pred, sdf_gt):
    """
    Loss matching the paper's description:
    - L1 loss between predicted and ground-truth signed distances.

    Returns:
        total_loss (tensor), dict with components
    """
    # Ensure shapes: [B, N] or [B, N, 1] -> squeeze last dim
    sdf_pred_clamped = sdf_pred
    sdf_gt_clamped = sdf_gt

    # L1 loss per the paper
    loss_l1 = F.l1_loss(sdf_pred_clamped.squeeze(), sdf_gt_clamped.squeeze(), reduction='mean')

    return loss_l1, {"l1": loss_l1.item()}

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

def train_stage_1_vae(vae_module, train_dataloader, optimizer, device, num_epochs, ckpt_dir="checkpoints_vae", resume=False, beta_kl=1e-5, prior_std=0.25):
    """
    Stage 1: train VAE. Keeps reconstruction loss but uses KL that regularizes
    q(z|π) toward N(0, prior_std^2) and weights KL by beta_kl (default 1e-5 per paragraph).
    """
    vae_module.train()
    scaler = GradScaler()
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch = 0

    ckpt_path = os.path.join(ckpt_dir, "vae_last.pth")
    if resume and os.path.exists(ckpt_path):
        start_epoch, _ = load_checkpoint(vae_module, optimizer, ckpt_path, device)

    for epoch in range(start_epoch, num_epochs):
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        for batch_idx, (point_clouds, query_points, _) in enumerate(train_dataloader):
            point_clouds = torch.stack(point_clouds).to(device)
            optimizer.zero_grad()

            with autocast():
                # VAE forward pass: returns (x_recon, z, latent_pc, mu, logvar)
                x_recon, z, latent_pc, mu, logvar = vae_module(point_clouds)
                # Reconstruction loss: keep for VAE training
                recon_loss = F.mse_loss(x_recon, latent_pc, reduction='mean')
                
                # KL divergence to N(0, prior_std^2)
                sigma2 = logvar.exp()
                prior_var = prior_std ** 2
                # KL per-sample
                kl_per_sample = 0.5 * ( (sigma2 + mu.pow(2)) / prior_var - 1 - logvar + math.log(prior_var) ).sum(dim=1)
                kl = kl_per_sample.mean()
                
                # Total VAE loss with small beta as described
                loss = recon_loss + beta_kl * kl

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(vae_module.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_recon_loss += recon_loss.item()
            total_kl_loss += kl.item()

        avg_recon_loss = total_recon_loss / len(train_dataloader)
        avg_kl_loss = total_kl_loss / len(train_dataloader)
        avg_total_loss = avg_recon_loss + beta_kl * avg_kl_loss
        
        print(f"Stage 1 - Epoch [{epoch + 1}/{num_epochs}], "
              f"Total Loss: {avg_total_loss:.6f}, "
              f"Recon Loss: {avg_recon_loss:.6f}, "
              f"KL Loss: {avg_kl_loss:.6f}, "
              f"Beta: {beta_kl:.6e}")
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            checkpoint_loss = {
                'total': avg_total_loss,
                'recon': avg_recon_loss,
                'kl': avg_kl_loss,
                'beta': beta_kl
            }
            save_checkpoint(vae_module, optimizer, epoch, checkpoint_loss, os.path.join(ckpt_dir, "vae_last.pth"))
            save_checkpoint(vae_module, optimizer, epoch, checkpoint_loss, os.path.join(ckpt_dir, f"vae_epoch_{epoch}.pth"))


def train_stage_2_modulation(modulation_module, train_dataloader, optimizer, device, num_epochs, ckpt_dir="checkpoints_mod", resume=False):
    """
    Stage 2: train modulation module for SDF prediction.
    Following the paragraph:
      - Use L1 loss between predicted and GT SDF for query points
      - Do NOT add a VAE reconstruction loss term here
      - KL regularization is part of the VAE (stage 1) and is already applied there
    """
    modulation_module.train()
    scaler = GradScaler()
    os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch = 0

    ckpt_path = os.path.join(ckpt_dir, "mod_last.pth")
    if resume and os.path.exists(ckpt_path):
        start_epoch, _ = load_checkpoint(modulation_module, optimizer, ckpt_path, device)

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        for batch_idx, (point_clouds, query_points, sdf_gt) in enumerate(train_dataloader):
            point_clouds = torch.stack(point_clouds).to(device)
            query_points = query_points.to(device)
            sdf_gt = sdf_gt.to(device)

            optimizer.zero_grad()

            with autocast():
                # modulation_module returns (sdf_pred, z, latent_pc, x_recon, mu, logvar) or similar
                outputs = modulation_module(point_clouds, query_points)
                sdf_pred = outputs[0]
                # other outputs are available if needed
                if sdf_pred.dim() == 3:
                    sdf_pred = sdf_pred.squeeze(-1)

                # Per-paper SDF loss (L1)
                loss, loss_dict = sdf_loss_function(sdf_pred, sdf_gt)
                
                # IMPORTANT: do NOT add a VAE reconstruction loss here (per the paragraph)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Stage 2 - Epoch [{epoch + 1}/{num_epochs}] - Total Loss: {avg_loss:.6f}")

        if (epoch + 1) % 100 == 0:
            save_checkpoint(modulation_module, optimizer, epoch, avg_loss, os.path.join(ckpt_dir, "mod_last.pth"))
            save_checkpoint(modulation_module, optimizer, epoch, avg_loss, os.path.join(ckpt_dir, f"mod_epoch_{epoch}.pth"))



# def staged_training(modulation_module, train_dataloader, device, num_epochs_stage_1, num_epochs_stage_2):
#     """
#     Full staged training pipeline.
#     """
def staged_training(modulation_module, train_dataloader, device, num_epochs_stage_1, num_epochs_stage_2, beta_kl=1e-5, prior_std=0.25, lr=1e-4):
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

                sdf_loss, loss_dict = sdf_loss_function(sdf_pred, sdf_gt)

                # KL loss (if mu/logvar available)
                if (mu is not None) and (logvar is not None):
                    sigma2 = logvar.exp()
                    prior_var = prior_std ** 2
                    kl_per_sample = 0.5 * ((sigma2 + mu.pow(2)) / prior_var - 1 - logvar + math.log(prior_var)).sum(dim=1)
                    kl = kl_per_sample.mean()
                else:
                    kl = torch.tensor(0.0, device=sdf_loss.device)

                # Combined loss: SDF + beta * KL
                loss = sdf_loss + beta_kl * kl 

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

        avg_sdf = epoch_sdf_loss / len(train_dataloader)
        avg_kl = epoch_kl_loss / len(train_dataloader)
        avg_recon = epoch_recon_loss / len(train_dataloader)

        print(f"Epoch [{epoch + 1}/{total_epochs}] - SDF: {avg_sdf:.6f}, KL: {avg_kl:.6f}, Recon(logged): {avg_recon:.6f}")

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
    encoding_dim = 512
    latent_dim = 4096
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 1000
    learning_rate = 1e-3
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
    dataset = VoxelSDFDataset(voxel_grids, num_query_points=20000,fixed_surface_points_size=20000, noise_std=0.0, device=device)
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    # modulation_module = ModulationModule(pointnet_input_dim=3, pointnet_output_dim=256, latent_dim=128).to(device)
    vae = ImprovedVAE(input_dim=latent_dim, latent_dim=encoding_dim, hidden_dim=1024, num_layers=8).to(device)
    sdf_network = ImprovedSDFNetwork(input_dim=encoding_dim, latent_dim = latent_dim, hidden_dim=512, output_dim=1, num_layers=8).to(device)
    modulation_module = ModulationModule(vae, sdf_network).to(device)
    optimizer = torch.optim.Adam(modulation_module.parameters(), lr=1e-4)
    staged_training(modulation_module, train_dataloader, device, num_epochs_stage_1=25, num_epochs_stage_2=10000)
    torch.save(modulation_module.state_dict(), "modulation_module.pth")

if __name__ == "__main__":
    torch.cuda.empty_cache()  # After each step
    main()