import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import norm
from utils.preprocess_data import *
from trainer import *
from models import *

def sample_from_diffusion_model(diffusion_model, modulation_module, device, timesteps, betas, alphas_cumprod, num_samples=1):
    """
    Sample from the diffusion model by iteratively denoising random noise.
    """
    diffusion_model.eval()  # Set to evaluation mode
    modulation_module.eval()  # Set to evaluation mode
    
    # Start with random noise
    z_t = torch.randn((num_samples, diffusion_model.layers[0][0].in_features), device=device)
    
    # Iteratively denoise
    for t in reversed(range(timesteps)):
        print(t)
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            predicted_noise = diffusion_model(z_t, t_tensor)
        
        # Reverse process: remove noise
        sqrt_recip_alphas_t = torch.sqrt(1.0 / alphas[t])
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod[t])
        z_t = sqrt_recip_alphas_t * (z_t - predicted_noise * betas[t] / sqrt_one_minus_alphas_cumprod_t)
        
        if t > 0:
            noise = torch.randn_like(z_t)
            z_t += torch.sqrt(betas[t]) * noise
    
    return z_t

def decode_latent_sample(modulation_module, z, query_points, device):
    """
    Decode the latent sample `z` into a point cloud using the modulation module.
    """
    print("Decoding...")
    modulation_module.eval()  # Set to evaluation mode
    
    with torch.no_grad():
        # # Expand `z` to match the number of query points
        z_expanded = z.unsqueeze(1).expand(-1, query_points.size(1), -1)
        
        # # Concatenate query points with the latent sample
        concatenated_input = torch.cat([query_points.to(device), z_expanded], dim=2)
        
        # Forward pass through the SDF network
        sdf_values = modulation_module.sdf_network(concatenated_input)
    print("Decoded!")
    return sdf_values

import numpy as np
from skimage.measure import marching_cubes

def extract_surface(sdf_values, resolution=64):
    """
    Extract a 3D surface from SDF values using Marching Cubes.
    """
    print("Extracting...")
    # Reshape SDF values into a 3D grid
    sdf_grid = sdf_values.view(resolution, resolution, resolution).cpu().numpy()
    
    # Run Marching Cubes
    vertices, faces, _, _ = marching_cubes(sdf_grid, level=0)
    print("Extracted!")
    return vertices, faces

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def save_as_obj(vertices, faces, filename="output.obj"):
    """
    Save the extracted surface as a .obj file.
    """
    with open(filename, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")  # OBJ uses 1-based indexing

    print(f"Saved as {filename}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = ImprovedVAE(input_dim=256, latent_dim=128, hidden_dim=512, num_layers=4).to(device)
sdf_network = ImprovedSDFNetwork(input_dim=3 + 128, hidden_dim=256, output_dim=1, num_layers=8).to(device)
modulation_module = ModulationModule(vae, sdf_network).to(device)
diffusion_model = DiffusionModel(latent_dim=128, hidden_dim=512, num_layers=6).to(device)
modulation_module_checkpoint = torch.load("modulation_module.pth")
modulation_module.load_state_dict(modulation_module_checkpoint)  # No key access
diffusion_model_checkpoint = torch.load("diffusion_model.pth")
diffusion_model.load_state_dict(diffusion_model_checkpoint)  # No key access
# Parameters
num_samples = 1
resolution = 64
query_points = torch.rand((num_samples, resolution**3, 3))*2-1  # Query points for SDF evaluation
num_epochs = 100
learning_rate = 1e-4
diffusion_steps = 100
# Example usage
timesteps = 100
betas = cosine_beta_schedule(timesteps).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
# Step 1: Sample from the diffusion model
z_t = sample_from_diffusion_model(diffusion_model, modulation_module, device, timesteps, betas, alphas_cumprod, num_samples)

# Step 2: Decode the latent sample into SDF values
sdf_values = decode_latent_sample(modulation_module, z_t, query_points, device)
print("SDF Min:", torch.min(sdf_values).item())
print("SDF Max:", torch.max(sdf_values).item())
# Step 3: Extract the 3D surface using Marching Cubes
vertices, faces = extract_surface(sdf_values, resolution)

# Step 4: Visualize the surface
save_as_obj(vertices, faces)