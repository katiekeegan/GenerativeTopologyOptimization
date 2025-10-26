import torch
import numpy as np
import os
import trimesh
from models import *
from utils.preprocess_data import create_voxel_grids
from dl4to.datasets import SELTODataset
from skimage import measure  # for marching cubes
from utils.preprocess_data import *
from torch.utils.data import DataLoader, Dataset

import torch
import numpy as np
from scipy.interpolate import griddata

from scipy.spatial import cKDTree

@torch.no_grad()
def extract_shape_from_latent(modulation_module, dataloader, device, voxel_grid, filename="extracted_shape.obj", grid_resolution=128):
    """
    Extract and save a shape as an .obj file using a latent point cloud and a dense 3D grid of query points.
    The query grid is normalized to match the [-1, 1]^3 space used during training.
    """
    modulation_module.eval()

    # 1. Get one batch of point clouds
    point_cloud, _, _ = next(iter(dataloader))
    point_cloud = point_cloud[0].unsqueeze(0).to(device)  # [1, N, 3]
    del dataloader

    # 2. Save input voxel mesh
    voxel_np = voxel_grid.squeeze().cpu().numpy()
    print(f"Voxel grid range: min={voxel_np.min():.4f}, max={voxel_np.max():.4f}")
    if voxel_np.min() < 0.5 < voxel_np.max():
        verts_gt, faces_gt, normals_gt, _ = measure.marching_cubes(voxel_np, level=0.5, spacing=(1.0 / voxel_np.shape[0],) * 3)
        verts_gt = verts_gt / voxel_np.shape[0]
        mesh_gt = trimesh.Trimesh(vertices=verts_gt, faces=faces_gt, process=False)
        mesh_gt.rezero()
        mesh_gt.fix_normals()
        mesh_gt.export("input_surface.obj")
        print("Input surface mesh saved to: input_surface.obj")
    else:
        print("Skipping input_surface.obj: level not within voxel range.")

    # 3. Generate padded 3D query grid in [0, 1]^3
    lin = torch.linspace(0.0, 1.0, grid_resolution, device=device)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), dim=-1)  # [R, R, R, 3]
    query_points = grid.reshape(-1, 3).unsqueeze(0)  # [1, R^3, 3]

    # 4. Normalize query points to match training: [0,1]^3 → [-1,1]^3
    query_points = (query_points - 0.5) * 2.0

    # 5. Predict SDF
    outputs = modulation_module(point_cloud, query_points)
    sdf_pred = outputs[0]  # First element is sdf_pred
    sdf_pred = sdf_pred.view(grid_resolution, grid_resolution, grid_resolution).cpu().numpy()
    print(f"SDF pred stats: min={sdf_pred.min():.4f}, max={sdf_pred.max():.4f}, mean={sdf_pred.mean():.4f}")

    if not (sdf_pred.min() < 0.0 < sdf_pred.max()):
        raise ValueError("SDF predictions do not cross zero — cannot extract surface.")

    spacing = 2.0 / grid_resolution
    verts, faces, normals, _ = measure.marching_cubes(sdf_pred, level=0.0, spacing=(spacing,) * 3)
    verts = verts - 1.0  # map to [-1,1]^3

    # 8. Build mesh and (optionally) clean it
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.rezero()
    mesh.fix_normals()

    # Optionally keep largest component
    # components = mesh.split(only_watertight=False)
    # print(f"Found {len(components)} mesh components.")
    # mesh = max(components, key=lambda m: m.area)

    # 9. Save mesh
    mesh.apply_scale(100.0)
    mesh.export(filename)
    print(f"Mesh saved to: {filename}")


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and a single point cloud
    selto = SELTODataset(root='.', name='sphere_complex', train=True)
    _, voxel_field = selto[0]  # Tuple: (problem, solution)
    voxel_grid = voxel_field.θ.squeeze().unsqueeze(0)  # shape: [1, D, H, W]
    dataset = VoxelSDFDataset(voxel_grid, num_query_points=20000, fixed_surface_points_size=20000, noise_std=0.05, device=device)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # Load model
    encoding_dim = 512
    latent_dim = 4096
    vae = ImprovedVAE(input_dim=latent_dim, latent_dim=encoding_dim, hidden_dim=1024, num_layers=4).to(device)
    sdf_network = ImprovedSDFNetwork(input_dim=encoding_dim, latent_dim=latent_dim, hidden_dim=512, output_dim=1, num_layers=4).to(device)
    modulation_module = ModulationModule(vae, sdf_network).to(device)

    checkpoint = torch.load("checkpoints_mod/mod_last.pth", map_location=device)
    if "model_state_dict" in checkpoint:
        modulation_module.load_state_dict(checkpoint["model_state_dict"])
    else:
        modulation_module.load_state_dict(checkpoint)

    # Run extraction with voxel grid passed in
    extract_shape_from_latent(modulation_module, train_dataloader, device, voxel_grid=voxel_grid, filename="output_shape.obj")
if __name__ == "__main__":
    main()