import os
import trimesh
import numpy as np
import pandas as pd
import random
import json
from scipy.spatial import cKDTree

# Define paths
raw_mesh_folder = '../data/SELTO/'  # Raw mesh directory
output_folder = '../Diffusion-SDF/data/acronym/SELTO/'  # Folder to store processed meshes (SDF data)
grid_output_folder = '../Diffusion-SDF/data/grid_data/acronym/SELTO/'  # Folder to store grid samples
splits_folder = '../Diffusion-SDF/data/splits/'  # Folder to store split data
split_filename = 'selto_all.json'  # Split filename

def compute_sdf(mesh, query_points):
    """
    Compute signed distance field (SDF) for each query point.
    The SDF is computed as the distance to the mesh surface.
    """
    # Convert mesh to trimesh object
    tree = cKDTree(mesh.vertices)
    distances, _ = tree.query(query_points)  # Get distances
    return distances

def sample_near_surface(mesh, num_samples=7000, sigma=0.05):
    """
    Sample points near the surface of the mesh.
    """
    # Sample points on the mesh surface
    surface_points, _ = trimesh.sample.sample_surface(mesh, num_samples)
    # Add Gaussian noise to simulate near-surface sampling
    noise = np.random.normal(0, sigma, size=surface_points.shape)
    near_surface_points = surface_points + noise
    return near_surface_points

def sample_uniform_grid(num_samples=3000):
    """
    Uniformly sample points in the 3D grid space (-1, -1, -1) to (1, 1, 1).
    """
    grid_samples = np.random.uniform(-1, 1, size=(num_samples, 3))
    return grid_samples

def save_sdf_csv(sdf_values, query_points, output_path):
    """
    Save the SDF values and query points to a CSV file, ensuring numerical consistency.
    """
    df = pd.DataFrame(np.hstack((query_points, sdf_values.reshape(-1, 1))),
                      columns=["x", "y", "z", "signed_distance"])
    
    # Convert all columns to float (ensuring no string or object types exist)
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Drop any rows with NaN values that might have been introduced
    df.dropna(inplace=True)

    df.to_csv(output_path, index=False, header=False)  # Ensure no headers (matches training format)
    print(f"Saved {output_path}")

def process_all_meshes():
    """
    Process all meshes in the raw_mesh folder and store the SDF data in CSV files.
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(grid_output_folder, exist_ok=True)

    for mesh_file in os.listdir(raw_mesh_folder):
        if mesh_file.endswith(".obj"):
            # Load mesh
            mesh_path = os.path.join(raw_mesh_folder, mesh_file)
            mesh = trimesh.load(mesh_path)
            model_idx = os.path.splitext(mesh_file)[0]  # Get model ID from filename

            # Create output subfolders for this model
            model_output_folder = os.path.join(output_folder, model_idx)
            grid_model_output_folder = os.path.join(grid_output_folder, model_idx)
            os.makedirs(model_output_folder, exist_ok=True)
            os.makedirs(grid_model_output_folder, exist_ok=True)

            # Sample near-surface points and compute SDF
            near_surface_points = sample_near_surface(mesh, num_samples=7000)
            near_surface_sdf = compute_sdf(mesh, near_surface_points)
            near_surface_output_path = os.path.join(model_output_folder, "sdf_data.csv")
            save_sdf_csv(near_surface_sdf, near_surface_points, near_surface_output_path)

            # Sample uniform grid points and compute SDF
            uniform_grid_points = sample_uniform_grid(num_samples=3000)
            uniform_grid_sdf = compute_sdf(mesh, uniform_grid_points)
            grid_output_path = os.path.join(grid_model_output_folder, "grid_gt.csv")
            save_sdf_csv(uniform_grid_sdf, uniform_grid_points, grid_output_path)

def generate_splits(train_ratio=0.8):
    """
    Generate train-test splits randomly for the dataset of meshes.
    By default, 80% of the meshes will be used for training and 20% for testing.
    """
    # Get all mesh filenames
    mesh_files = [f for f in os.listdir(raw_mesh_folder) if f.endswith(".obj")]

    # Create the splits dictionary with numerical indices as mesh IDs
    splits = {
        "acronym": {
            "SELTO": 
                [str(idx) for idx in range(len(mesh_files))]            
        }
    }

    # Ensure the splits folder exists
    os.makedirs(splits_folder, exist_ok=True)

    # Save the splits as a JSON file
    with open(os.path.join(splits_folder, split_filename), 'w') as f:
        json.dump(splits, f, indent=4)
    print(f"Saved splits to {os.path.join(splits_folder, split_filename)}")

if __name__ == "__main__":
    process_all_meshes()
    generate_splits()