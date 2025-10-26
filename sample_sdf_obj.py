import os
import argparse
import torch
import numpy as np
import trimesh
from skimage import measure

from models import ImprovedVAE, ImprovedSDFNetwork, ModulationModule


def infer_latent_dim(modulation_module, ckpt=None):
    """
    Robustly infer the latent dimension (kept for fallback).
    """
    # 1) direct attribute on vae
    if hasattr(modulation_module, "vae"):
        vae = modulation_module.vae
        if hasattr(vae, "latent_dim") and isinstance(getattr(vae, "latent_dim"), int):
            return int(vae.latent_dim)
        if hasattr(vae, "fc_mu"):
            fc_mu = vae.fc_mu
            if hasattr(fc_mu, "out_features"):
                return int(fc_mu.out_features)
            if hasattr(fc_mu, "weight") and hasattr(fc_mu.weight, "shape"):
                return int(fc_mu.weight.shape[0])

    # 2) sdf_network input_dim
    if hasattr(modulation_module, "sdf_network"):
        sdf_net = modulation_module.sdf_network
        if hasattr(sdf_net, "input_dim") and isinstance(getattr(sdf_net, "input_dim"), int):
            return int(sdf_net.input_dim)

    # 3) try state_dict on the live module
    try:
        sd = modulation_module.state_dict()
        for k, v in sd.items():
            if "vae" in k and ("fc_mu" in k or "fc_mu.weight" in k):
                try:
                    return int(v.shape[0])
                except Exception:
                    pass
    except Exception:
        pass

    # 4) try checkpoint dict if provided
    if ckpt is not None:
        if isinstance(ckpt, dict):
            sd = None
            if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
                sd = ckpt["model_state_dict"]
            elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                sd = ckpt
            if sd is not None:
                for k, v in sd.items():
                    if "vae" in k and ("fc_mu" in k or "fc_mu.weight" in k):
                        try:
                            return int(v.shape[0])
                        except Exception:
                            pass

    raise RuntimeError("Cannot infer latent dimension from modulation_module or checkpoint; please provide modulation_module.vae.latent_dim or modulation_module.sdf_network.input_dim")


@torch.no_grad()
def sample_sdf_from_prior_and_save(modulation_module, device, filename="sampled_shape.obj",
                                   grid_resolution=128, prior_sigma=0.25,
                                   latent_dim=None, ckpt=None):
    """
    Sample z ~ N(0, prior_sigma^2), evaluate SDF network on dense grid and save .obj.
    If latent_dim is provided, use it directly (recommended). Otherwise try to infer.
    """
    modulation_module.eval()

    # Build query grid in [-1,1]^3 (matching training normalization)
    lin = torch.linspace(-1.0, 1.0, grid_resolution, device=device)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), dim=-1)  # [R, R, R, 3]
    query_points = grid.reshape(-1, 3).unsqueeze(0)  # [1, R^3, 3]

    # Determine latent dimension
    if latent_dim is None:
        latent_dim = infer_latent_dim(modulation_module, ckpt=ckpt)
    else:
        latent_dim = int(latent_dim)

    # Sample z from prior N(0, prior_sigma^2)
    z = torch.randn(1, latent_dim, device=device) * float(prior_sigma)  # [1, latent_dim]

    # Evaluate SDF network. Prefer calling sdf_network directly to avoid needing a point cloud.
    if hasattr(modulation_module, "sdf_network"):
        sdf_pred = modulation_module.sdf_network(z, query_points)  # expected shape [1, R^3, 1] or [1, R^3]
    else:
        # fallback: try calling modulation_module with a dummy point cloud if sdf_network not available
        dummy_pc = torch.zeros(1, 1, 3, device=device)
        outputs = modulation_module(dummy_pc, query_points)
        if isinstance(outputs, (list, tuple)):
            sdf_pred = outputs[0]
        else:
            raise RuntimeError("modulation_module returned unexpected output; cannot extract sdf_pred")

    # Normalize/reshape predicted SDF to [R,R,R] numpy array
    sdf_t = sdf_pred.detach().cpu()
    # collapse batch dim if present
    if sdf_t.dim() == 3 and sdf_t.size(0) == 1:
        sdf_t = sdf_t.squeeze(0)
    if sdf_t.dim() == 2 and sdf_t.size(-1) == 1:
        sdf_t = sdf_t.squeeze(-1)

    # At this point sdf_t should be shape [R^3] or already [R,R,R]
    if sdf_t.dim() == 1:
        try:
            sdf_grid = sdf_t.view(grid_resolution, grid_resolution, grid_resolution).numpy()
        except Exception as e:
            raise RuntimeError(f"Could not reshape SDF predictions (len={sdf_t.shape[0]}) into ({grid_resolution}^3): {e}")
    elif sdf_t.dim() == 3:
        sdf_grid = sdf_t.numpy()
    else:
        sdf_flat = sdf_t.reshape(-1)
        try:
            sdf_grid = sdf_flat.view(grid_resolution, grid_resolution, grid_resolution).numpy()
        except Exception as e:
            raise RuntimeError(f"Unexpected SDF tensor shape {sdf_t.shape}; cannot reshape to grid: {e}")

    # Basic checks
    print(f"SDF pred stats: min={sdf_grid.min():.6f}, max={sdf_grid.max():.6f}, mean={sdf_grid.mean():.6f}")
    if not (sdf_grid.min() < 0.0 < sdf_grid.max()):
        raise ValueError("SDF predictions do not cross zero — cannot extract surface. min/max: {:.6f}/{:.6f}".format(sdf_grid.min(), sdf_grid.max()))

    # Extract mesh with marching cubes.
    # marching_cubes with spacing=(1,1,1) returns vertex coordinates in index space [0..R-1].
    verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=0.0, spacing=(1.0, 1.0, 1.0))
    # Map index coordinates [0, R-1] -> [-1,1]
    verts_norm = verts / float(max(1, (grid_resolution - 1))) * 2.0 - 1.0

    # Build mesh and save
    mesh = trimesh.Trimesh(vertices=verts_norm, faces=faces, process=False)
    mesh.rezero()
    mesh.fix_normals()

    mesh.apply_scale(100.0)  # scale to printable units (optional)
    mesh.export(filename)
    print(f"Saved sampled mesh to: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Sample SDF from VAE prior and export a mesh (.obj)")
    parser.add_argument("--encoding-dim", type=int, default=512, help="encoding dimension (vae latent vector size)")
    parser.add_argument("--latent-dim", type=int, default=4096, help="VAE input / flattened latent dimension")
    parser.add_argument("--grid", type=int, default=128, help="grid resolution (R). R^3 queries will be evaluated")
    parser.add_argument("--outfile", type=str, default="sampled_shape.obj", help="output OBJ filename")
    parser.add_argument("--prior-sigma", type=float, default=0.25, help="prior standard deviation (default 0.25)")
    parser.add_argument("--ckpt", type=str, default="checkpoints_mod/mod_last.pth", help="modulation module checkpoint path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build models with the provided dims
    encoding_dim = int(args.encoding_dim)
    latent_dim = int(args.latent_dim)
    vae = ImprovedVAE(input_dim=latent_dim, latent_dim=encoding_dim, hidden_dim=1024, num_layers=4).to(device)
    sdf_network = ImprovedSDFNetwork(input_dim=encoding_dim, latent_dim=latent_dim, hidden_dim=512, output_dim=1, num_layers=4).to(device)
    modulation_module = ModulationModule(vae, sdf_network).to(device)

    # Load modulation module checkpoint (robust)
    ckpt = None
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        try:
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                modulation_module.load_state_dict(ckpt["model_state_dict"])
                print(f"Loaded modulation module state_dict from {args.ckpt}")
            else:
                modulation_module.load_state_dict(ckpt)
                print(f"Loaded modulation module state_dict (raw) from {args.ckpt}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint into modulation_module: {e}. Proceeding with current init.")
    else:
        print(f"No checkpoint found at {args.ckpt}; proceeding with randomly initialized model (not recommended).")

    # Use provided latent_dim to avoid inference issues
    sample_sdf_from_prior_and_save(modulation_module, device,
                                   filename=args.outfile,
                                   grid_resolution=int(args.grid),
                                   prior_sigma=float(args.prior_sigma),
                                   latent_dim=latent_dim,
                                   ckpt=ckpt)


if __name__ == "__main__":
    main()