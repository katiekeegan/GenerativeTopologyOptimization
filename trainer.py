import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from models import *
from utils.preprocess_data import *
from dl4to.datasets import SELTODataset

def compute_sdf(query_points, surface_points, normals):
    """
    Compute the signed distance of each query point to the nearest surface point.

    Args:
        query_points (torch.Tensor): Tensor of shape (B, N, 3) containing query points.
        surface_points (torch.Tensor): Tensor of shape (B, M, 3) containing surface points.
        normals (torch.Tensor): Tensor of shape (B, M, 3) containing surface normals.

    Returns:
        sdf_values (torch.Tensor): Tensor of shape (B, N) containing signed distances.
    """
    # Compute pairwise distances between query points and surface points
    distances = torch.cdist(query_points, surface_points)  # Shape: (B, N, M)

    # Find the nearest surface point and its normal
    min_distances, min_indices = torch.min(distances, dim=2)  # Shape: (B, N)
    # Expand min_indices to match the last dimension of surface_points
    min_indices_expanded = min_indices.unsqueeze(-1).expand(-1, -1, 3)  # Shape: (B, N, 3)
    # breakpoint()
    # Use gather correctly
    nearest_surface_points = torch.gather(surface_points, 1, min_indices_expanded)
    nearest_normals = torch.gather(normals, 1, min_indices_expanded)

    # Compute inside-outside using dot product with surface normal
    vectors_to_surface = query_points - nearest_surface_points  # Shape: (B, N, 3)
    dot_product = torch.sum(vectors_to_surface * nearest_normals, dim=-1)  # Shape: (B, N)

    # Assign negative sign for inside points
    sdf_values = min_distances * torch.sign(dot_product)

    return sdf_values

def train_modulation_module(modulation_module, train_dataloader, optimizer, device, num_epochs):
    modulation_module.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (point_clouds, query_points, normals) in enumerate(train_dataloader):
            # point_clouds consists of strictly surface points
            # query_points consists of points both on and near the surface (used in the SDF network)
            # # Move point clouds to the device
            point_clouds = torch.stack(point_clouds).to(device)# point_clouds = torch.stack([pc.to(device) for pc in point_clouds]
            optimizer.zero_grad()
            # Forward pass
            recon_point_clouds, z, latent_pc, x_recon = modulation_module(point_clouds, query_points)
            query_points_sdf_values = compute_sdf(query_points, point_clouds, normals)
            reconstruction_loss = F.l1_loss(recon_point_clouds.squeeze(), query_points_sdf_values)
            vae_loss = F.mse_loss(x_recon, latent_pc)
            # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_batch_loss = reconstruction_loss # + kl_loss
            
            # Backward pass and optimization
            total_batch_loss.backward()
            optimizer.step()
            
            # Accumulate loss
            total_loss += total_batch_loss.item()
        
        # Print epoch loss
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], VAE Loss: {avg_loss:.4f}")

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
                sdf_values, z, mu, logvar = modulation_module(point_clouds, query_points)
            
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

def main():
    # train_modulation_module = True
    # Training hyperparameters
    latent_dim = 256
    encoding_dim = 64
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

    print("Loading SELTO dataset...")
    selto = SELTODataset(root='.', name='sphere_complex', train=True)
    print("SELTO dataset loaded!")
    print("Constructing voxel grids...")
    voxel_grids = create_voxel_grids(selto)
    print("Voxel grids constructed!")

    # Create the dataset and DataLoader
    dataset = VoxelSDFDataset(voxel_grids, num_query_points=10000, noise_std=0.5, device=device)
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    # modulation_module = ModulationModule(pointnet_input_dim=3, pointnet_output_dim=256, latent_dim=128).to(device)
    vae = ImprovedVAE(input_dim=latent_dim, latent_dim=encoding_dim, hidden_dim=512, num_layers=4).to(device)
    sdf_network = ImprovedSDFNetwork(input_dim=3 + latent_dim, latent_dim = encoding_dim, hidden_dim=128, output_dim=1, num_layers=8).to(device)
    modulation_module = ModulationModule(vae, sdf_network).to(device)
    optimizer = torch.optim.Adam(modulation_module.parameters(), lr=1e-4)
    train_modulation_module(modulation_module, train_dataloader, optimizer, device, num_epochs=num_epochs)
    torch.save(modulation_module.state_dict(), "modulation_module.pth")
    # modulation_module_checkpoint = torch.load("modulation_module.pth")
    # modulation_module.load_state_dict(modulation_module_checkpoint)  # No key access

    print("Training diffusion model...")
    diffusion_model = DiffusionModel(latent_dim=128, hidden_dim=512, num_layers=6, timesteps=diffusion_steps).to(device)
    diffusion_optimizer = optim.Adam(diffusion_model.parameters(), lr=learning_rate)
    train_diffusion_model(diffusion_model, modulation_module, train_dataloader, diffusion_optimizer, device, diffusion_steps, betas, alphas_cumprod, 1000)
    print("Diffusion model trained!")

    # After training the modulation module
    torch.save(diffusion_model.state_dict(), "diffusion_model.pth")

if __name__ == "__main__":
    main()