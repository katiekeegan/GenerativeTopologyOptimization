import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from dl4to.datasets import SELTODataset

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation

import trimesh

from skimage import measure

import os

def repair_mesh(mesh):
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fill_holes()
    mesh.remove_infinite_values()
    mesh.rezero()

    if not mesh.is_winding_consistent:
        mesh.fix_normals()

    return mesh
class VoxelSDFDataset(Dataset):
    def __init__(self, voxel_grids, num_query_points=1000, noise_std=0.05, 
                 fixed_surface_points_size=2000, device='cpu', surface_sample_ratio=0.7):
        self.voxel_grids = voxel_grids
        self.num_query_points = num_query_points
        self.noise_std = noise_std
        self.fixed_surface_points_size = fixed_surface_points_size
        self.device = device
        self.surface_sample_ratio = surface_sample_ratio

        self.surface_data = []
        self.sdf_grids = []

        # Compute shared normalization constants
        D, H, W = voxel_grids[0].shape  # e.g., (39, 39, 21)
        self.voxel_dims = torch.tensor([D, H, W], dtype=torch.float32)
        self.center = (self.voxel_dims - 1) / 2.0
        self.scale = self.voxel_dims.max() / 2.0  # To map longest side to [-1,1]

        for vg in voxel_grids:
            surface_points, normals = self._precompute_surface(vg)
            self.surface_data.append((surface_points, normals))
            self.sdf_grids.append(self._get_signed_distance_grid(vg.numpy()))

    def _precompute_surface(self, voxel_grid):
        voxel_np = voxel_grid.numpy()
        try:
            verts, faces, _, _ = measure.marching_cubes(voxel_np, level=0.5)
        except ValueError:
            return torch.empty((0, 3)), torch.empty((0, 3))

        # Normalize verts into [-1, 1]^3
        verts = torch.tensor(verts.copy(), dtype=torch.float32)
        verts_normalized = (verts - self.center) / self.scale

        mesh = trimesh.Trimesh(vertices=verts_normalized.numpy(), faces=faces, process=False)
        N = self.fixed_surface_points_size * 2
        surface_points, face_indices = mesh.sample(N, return_index=True)
        normals = mesh.face_normals[face_indices]
        return torch.tensor(surface_points, dtype=torch.float32), torch.tensor(normals, dtype=torch.float32)

    def _get_signed_distance_grid(self, voxel_np):
        voxel_bool = voxel_np.astype(bool)
        outside = distance_transform_edt(~voxel_bool)
        inside = distance_transform_edt(voxel_bool)
        sdf = outside - inside
        return torch.tensor(sdf, dtype=torch.float32)

    def __len__(self):
        return len(self.voxel_grids)

    def __getitem__(self, idx):
        surface_points, normals = self.surface_data[idx]
        surface_points, normals = self._process_to_fixed_size(surface_points, normals)

        query_points = self._generate_query_points(surface_points).to(self.device)

        # Prepare for grid_sample: query_points ∈ [-1,1]
        sdf_grid = self.sdf_grids[idx].unsqueeze(0).unsqueeze(0).to(self.device)
        coords = query_points.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, N, 1, 1, 3]
        sdf_values = F.grid_sample(sdf_grid, coords, align_corners=True).squeeze()

        # Normalize SDF by geometric scale
        sdf_values /= self.scale

        return surface_points.to(self.device), query_points, sdf_values.to(self.device)

    def _process_to_fixed_size(self, points, normals):
        n_points = points.shape[0]
        if n_points == self.fixed_surface_points_size:
            return points, normals
        elif n_points < self.fixed_surface_points_size:
            indices = torch.randint(0, n_points, (self.fixed_surface_points_size,))
            return points[indices], normals[indices]
        else:
            idx = torch.randperm(n_points)[:self.fixed_surface_points_size]
            return points[idx], normals[idx]

    def _generate_query_points(self, surface_points):
        device = surface_points.device
        n_surface = int(self.num_query_points * self.surface_sample_ratio)
        n_space = self.num_query_points - n_surface
        eps = 0.0

        if len(surface_points) > 0:
            idx = torch.randint(0, len(surface_points), (n_surface,), device=device)
            surface_samples = surface_points[idx] + torch.randn(n_surface, 3, device=device) * self.noise_std
        else:
            surface_samples = torch.empty(n_surface, 3, device=device).uniform_(-1, 1)

        surface_samples = surface_samples.clamp(-1 - eps, 1 + eps)
        space_samples = torch.empty(n_space, 3, device=device).uniform_(-1 - eps, 1 + eps)
        query_points = torch.cat([surface_samples, space_samples], dim=0)[torch.randperm(self.num_query_points)]
        return query_points


def collate_fn(self, batch):
    # Ensure we're working with tuples
    batch = [item if isinstance(item, tuple) else (item['surface_points'], item['query_points'], item['sdf_values']) 
             for item in batch]
    
    point_clouds = [item[0] for item in batch]
    query_points = torch.stack([item[1] for item in batch])
    sdf_values = torch.stack([item[2] for item in batch])
    return point_clouds, query_points, sdf_values

    def _process_to_fixed_size(self, points, normals):
        n_points = points.shape[0]
        if n_points == self.fixed_surface_points_size:
            return points, normals
        if n_points < self.fixed_surface_points_size:
            repeat = (self.fixed_surface_points_size // n_points) + 1
            points = points.repeat(repeat, 1)
            normals = normals.repeat(repeat, 1)
        indices = torch.randperm(points.shape[0])[:self.fixed_surface_points_size]
        return points[indices], normals[indices]

    def _generate_query_points(self, surface_points):
        device = surface_points.device
        n_surface = int(self.num_query_points * self.surface_sample_ratio)
        n_space = self.num_query_points - n_surface
        eps = 0.0  # padding beyond [-1, 1]^3

        if len(surface_points) > 0:
            idx = torch.randint(0, len(surface_points), (n_surface,), device=device)
            surface_samples = surface_points[idx] + torch.randn(n_surface, 3, device=device) * self.noise_std
        else:
            surface_samples = torch.empty(n_surface, 3, device=device).uniform_(-1, 1)

        surface_samples = surface_samples.clamp(-1 - eps, 1 + eps)
        space_samples = torch.rand((n_space, 3), device=device) * (2 + 2*eps) - (1 + eps)
        query_points = torch.cat([surface_samples, space_samples], dim=0)
        return query_points[torch.randperm(self.num_query_points)]

def create_voxel_grids(dataset):
    voxel_grids = []
    for model_idx in range(len(dataset)):
        problem, solution = dataset[model_idx]
        density = solution.θ.squeeze().cpu()
        voxel_grids.append(density)
        del density
    return voxel_grids

def collate_fn(batch):
    point_clouds = [item[0] for item in batch]
    query_points = torch.stack([item[1] for item in batch])
    sdf_values = torch.stack([item[2] for item in batch])
    return point_clouds, query_points, sdf_values

