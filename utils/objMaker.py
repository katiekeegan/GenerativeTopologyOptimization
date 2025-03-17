import numpy as np
import os
import torch
import torch.nn.functional as F
from skimage import measure
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
from dl4to.datasets import SELTODataset
from skimage.measure import marching_cubes
import trimesh
import point_cloud_utils as pcu

# Resolution used to convert shapes to watertight manifolds
# Higher value means better quality and slower
manifold_resolution = 20_000

# Number of points in the volume to sample around the shape
num_vol_pts = 100_000

# Number of points on the surface to sample
num_surf_pts = 100_000

# Function to process a model
def process_model(dataset, model_idx, grid_resolution=(128, 128, 128), num_vol_pts=100000, num_surf_pts=100000):
    """
    Process a model from the dataset and save SDF data to file.

    Args:
        dataset (Dataset): Dataset containing the model.
        model_idx (int): Index of the model in the dataset.
        grid_resolution (tuple): Resolution of the voxel grid.
        num_vol_pts (int): Number of volume points to sample.
        num_surf_pts (int): Number of surface points to sample.
    """
    print(f"Processing model {model_idx}...")
    
    # Retrieve the model data
    problem, solution = dataset[model_idx]
    density = solution.θ.squeeze().cpu().numpy()  # Density or SDF tensor

    # Apply Marching Cubes to the binary density tensor to get vertices and faces
    verts, faces, _, _ = marching_cubes(density, level=0.5)  # level=0.5 works well for binary data

    # Create a mesh from vertices and faces using trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # Save SDF data
    path = os.path.join('../data', 'SELTO')
    os.makedirs(path, exist_ok=True)
    obj_path = os.path.join(f'{path}/{model_idx}.obj')
    # Export the mesh to an .obj file
    mesh.export(obj_path)
    print(f"Saved SDF for model {model_idx}")

    # v, f = pcu.load_mesh_vf(obj_path)

    # # Convert mesh to watertight manifold
    # vm, fm = pcu.make_mesh_watertight(v, f, manifold_resolution)
    # nm = pcu.estimate_mesh_vertex_normals(vm, fm)  # Compute vertex normals for watertight mesh

    # # Generate random points in the volume around the shape
    # # NOTE: ShapeNet shapes are normalized within [-0.5, 0.5]^3
    # p_vol = (np.random.rand(num_vol_pts, 3) - 0.5) * 1.1

    # # Comput the SDF of the random points
    # sdf, _, _  = pcu.signed_distance_to_mesh(p_vol, vm, fm)

    # # Sample points on the surface as face ids and barycentric coordinates
    # fid_surf, bc_surf = pcu.sample_mesh_random(vm, fm, num_surf_pts)

    # # Compute 3D coordinates and normals of surface samples
    # p_surf = pcu.interpolate_barycentric_coords(fm, fid_surf, bc_surf, vm)
    # n_surf = pcu.interpolate_barycentric_coords(fm, fid_surf, bc_surf, nm)

    # # Save volume points + SDF and surface points + normals
    # # Load using np.load()
    # npz_path = os.path.join(path, "samples.npz")
    # np.savez(npz_path, p_vol=p_vol, sdf_vol=sdf, p_surf=p_surf, n_surf=n_surf)
    # print(f"Saved SDF for model {model_idx}")

# Process all models
dataset = SELTODataset(root='../data/', name='sphere_complex', train=True)
for model_idx in range(len(dataset)):
    process_model(dataset, model_idx)