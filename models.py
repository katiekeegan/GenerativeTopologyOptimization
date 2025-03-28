import torch
import torch.nn as nn
import torch.nn.functional as F
class PointNetEncoder(nn.Module):
    
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=128):
        super(PointNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, 1)
        self.conv2 = nn.Conv1d(hidden_dim, output_dim, 1)
        self.bn1 = nn.LayerNorm(hidden_dim)  # LayerNorm for stability
        self.bn2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        # Input: (B, N, 3) -> Transpose to (B, 3, N) for Conv1d
        x = x.transpose(1, 2)  # Shape: (B, 3, N)

        # Apply Conv1d + LayerNorm + LeakyReLU
        x = self.conv1(x)  # Shape: (B, hidden_dim, N)
        x = x.transpose(1, 2)  # Shape: (B, N, hidden_dim)
        x = F.leaky_relu(self.bn1(x), 0.2)  # Apply LayerNorm on last dimension
        x = x.transpose(1, 2)  # Shape: (B, hidden_dim, N)

        # Second Conv1d + LayerNorm + LeakyReLU
        x = self.conv2(x)  # Shape: (B, output_dim, N)
        x = x.transpose(1, 2)  # Shape: (B, N, output_dim)
        x = F.leaky_relu(self.bn2(x), 0.2)  # Apply LayerNorm on last dimension
        x = x.transpose(1, 2)  # Shape: (B, output_dim, N)

        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]  # Shape: (B, output_dim, 1)
        x = x.view(-1, x.size(1))  # Shape: (B, output_dim)
        return x

class ImprovedVAE(nn.Module):
    def __init__(self, input_dim=128, latent_dim=64, hidden_dim=256, num_layers=4):
        super(ImprovedVAE, self).__init__()
        
        # PointNetEncoder for surface point clouds
        self.pointnetencoder = PointNetEncoder(output_dim=input_dim)

        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dim))
        encoder_layers.append(nn.LayerNorm(hidden_dim))  # LayerNorm
        encoder_layers.append(nn.LeakyReLU(0.2))
        
        for _ in range(num_layers - 1):
            encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            encoder_layers.append(nn.LayerNorm(hidden_dim))
            encoder_layers.append(nn.LeakyReLU(0.2))
        
        encoder_layers.append(nn.Linear(hidden_dim, latent_dim))  # Output latent vector directly
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (optional, for reconstruction)
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, hidden_dim))
        decoder_layers.append(nn.LayerNorm(hidden_dim))
        decoder_layers.append(nn.LeakyReLU(0.2))
        
        for _ in range(num_layers - 1):
            decoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            decoder_layers.append(nn.LayerNorm(hidden_dim))
            decoder_layers.append(nn.LeakyReLU(0.2))
        
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))  # Reconstruct global feature
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        # Encode surface point cloud
        latent_pc = self.pointnetencoder(x)
        z = self.encoder(latent_pc)  # Directly output latent vector
        
        # Decode (optional)
        x_recon = self.decoder(z)
        
        return x_recon, z, latent_pc  # Return reconstruction and latent vector

class ImprovedSDFNetwork(nn.Module):
    def __init__(self, input_dim=3, latent_dim=64, hidden_dim=256, output_dim=1, num_layers=8):
        super(ImprovedSDFNetwork, self).__init__()
        
        # Input layer (query points + latent code)
        self.fc_in = nn.Linear(input_dim, hidden_dim)

        # Hidden layers with skip connections
        self.fc_hidden = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        self.skip_connections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, z):
        """
        Args:
            x: Query points of shape (B, M, 3), where B is batch size and M is number of query points.
            z: Latent code of shape (B, latent_dim).
        Returns:
            sdf_values: Predicted SDF values of shape (B, M, 1).
        """
        batch_size, num_query_points, _ = x.shape

        # Expand latent code to match the number of query points
        z_expanded = z.unsqueeze(1).expand(-1, num_query_points, -1)  # Shape: (B, M, latent_dim)

        # Concatenate query points with latent code
        x = torch.cat([x, z_expanded], dim=-1)  # Shape: (B, M, 3 + latent_dim)

        # Reshape for Linear layer: (B * M, 3 + latent_dim)
        x = x.view(-1, x.size(-1))  # Shape: (B * M, 3 + latent_dim)

        # Input layer
        x = F.leaky_relu(self.fc_in(x), 0.2)  # Shape: (B * M, hidden_dim)

        # Hidden layers with skip connections
        for i, layer in enumerate(self.fc_hidden):
            x = F.leaky_relu(layer(x), 0.2)
            if i > 0:
                x = x + self.skip_connections[i - 1](x)  # Skip connection

        # Output layer with tanh activation
        x = self.fc_out(x)  # Shape: (B * M, 1)

        # Reshape back to (B, M, 1)
        x = x.view(batch_size, num_query_points, -1)  # Shape: (B, M, 1)
        return x

class ModulationModule(nn.Module):
    def __init__(self, vae, sdf_network):
        super(ModulationModule, self).__init__()
        self.vae = vae
        self.sdf_network = sdf_network

    def forward(self, point_cloud, query_points):
        # Step 1: Encode the surface point cloud
        x_recon, z, latent_pc = self.vae(point_cloud)

        # Step 2: Predict SDF values at query points
        sdf_values = self.sdf_network(query_points, x_recon)

        return sdf_values, z, latent_pc, x_recon

class FeatureWiseAttention(nn.Module):
    def __init__(self, latent_dim=128, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
    
    def forward(self, x):
        # Reshape from (batch, features) -> (batch, 1, features) to treat features as a sequence
        x = x.unsqueeze(1)  # Now shape is (batch, seq=1, features)
        x, _ = self.attn(x, x, x)  # Apply self-attention across features
        return x.squeeze(1)  # Restore shape to (batch, features)

class DiffusionModel(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=512, num_layers=6, timesteps=100):
        super().__init__()
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim),
            ) for _ in range(num_layers)
        ])
        
        self.feature_attn = FeatureWiseAttention(latent_dim, num_heads=2)  # Feature-wise attention

    def forward(self, z, t):
        t = t.float().unsqueeze(-1)  
        t_embed = self.time_embedding(t)  
        
        for layer in self.layers:
            z = layer(z + t_embed) + z  # Residual connection
        
        z = self.feature_attn(z)  # Apply feature-wise attention
        return z

import math

def linear_beta_schedule(timesteps, start=1e-4, end=2e-2):
    return torch.linspace(start, end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def forward_process(z, t, betas, alphas_cumprod):
    """
    Add noise to `z` at time step `t`.
    """
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[t])
    noise = torch.randn_like(z)  # Sample noise
    z_t = sqrt_alphas_cumprod_t.unsqueeze(-1) * z + sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1) * noise
    return z_t, noise

class DiffusionSDF(nn.Module):
    def __init__(self, modulation_module, diffusion_model):
        super(DiffusionSDF, self).__init__()
        self.modulation_module = modulation_module
        self.diffusion_model = diffusion_model

    def forward(self, point_cloud, query_points, t):
        sdf_values, z, mu, logvar = self.modulation_module(point_cloud, query_points)
        z_t = self.diffusion_model(z, t)
        return z_t, sdf_values, z, mu, logvar