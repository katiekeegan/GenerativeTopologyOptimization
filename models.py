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

        # Encoder - outputs intermediate representation
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dim))
        encoder_layers.append(nn.LayerNorm(hidden_dim))  # LayerNorm
        encoder_layers.append(nn.LeakyReLU(0.2))
        
        for _ in range(num_layers - 1):
            encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            encoder_layers.append(nn.LayerNorm(hidden_dim))
            encoder_layers.append(nn.LeakyReLU(0.2))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # VAE-specific: separate heads for mu and logvar
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

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
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + eps * std
        where eps ~ N(0, I) and std = exp(0.5 * logvar)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode surface point cloud
        latent_pc = self.pointnetencoder(x)
        h = self.encoder(latent_pc)  # Intermediate representation
        
        # VAE: split into mu and logvar
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Reparameterization trick to sample z
        z = self.reparameterize(mu, logvar)
        
        # Decode (optional)
        x_recon = self.decoder(z)
        
        # Return: (x_recon, z, latent_pc, mu, logvar)
        # - x_recon: reconstruction in same shape as latent_pc (will be used to reconstruct original input)
        # - z: sampled latent vector
        # - latent_pc: encoded point cloud features (for backward compatibility)
        # - mu, logvar: variational parameters for KL divergence computation
        return x_recon, z, latent_pc, mu, logvar
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class ImprovedSDFNetwork(nn.Module):
    def __init__(self, input_dim=3, point_dim=3,latent_dim=256, hidden_dim=256, num_blocks=4, output_dim=1, num_layers=4):
        super().__init__()

        # Input: [B, N, point_dim], latent: [B, latent_dim]
        self.input_proj = nn.Sequential(
            nn.Linear(point_dim + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )

        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),  # Output: [B, N, 1] (signed SDF)
            nn.Tanh()
        )

    def forward(self, query_points, latent):
        """
        Args:
            query_points: [B, N, 3]
            latent: [B, latent_dim]
        Returns:
            sdf_pred: [B, N]
        """
        B, N, _ = query_points.shape
        latent_expanded = latent.unsqueeze(1).expand(-1, N, -1)  # [B, N, latent_dim]
        x = torch.cat([query_points, latent_expanded], dim=-1)  # [B, N, point_dim + latent_dim]

        x = self.input_proj(x)
        x = self.res_blocks(x)
        sdf = self.output_head(x).squeeze(-1)  # [B, N]

        return sdf

class ModulationModule(nn.Module):
    def __init__(self, vae, sdf_network):
        super(ModulationModule, self).__init__()
        self.vae = vae
        self.sdf_network = sdf_network

    def forward(self, point_cloud, query_points):
        # Step 1: Encode the surface point cloud (now returns 5 values for VAE)
        x_recon, z, latent_pc, mu, logvar = self.vae(point_cloud)

        # Step 2: Predict SDF values at query points
        sdf_values = self.sdf_network(query_points, x_recon)

        # Return 6-tuple: (sdf_pred, z, latent_pc, x_recon, mu, logvar)
        return sdf_values, z, latent_pc, x_recon, mu, logvar

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
        
        # self.feature_attn = FeatureWiseAttention(latent_dim, num_heads=2)  # Feature-wise attention

    def forward(self, z, t):
        t = t.float().unsqueeze(-1)  
        t_embed = self.time_embedding(t)  
        
        for layer in self.layers:
            z = layer(z + t_embed) + z  # Residual connection
        
        # z = self.feature_attn(z)  # Apply feature-wise attention
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
        sdf_values, z, latent_pc, x_recon, mu, logvar = self.modulation_module(point_cloud, query_points)
        z_t = self.diffusion_model(z, t)
        return z_t, sdf_values, z, mu, logvar