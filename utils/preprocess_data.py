import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.ndimage import distance_transform_edt, binary_dilation
import trimesh
from skimage import measure
from dl4to.datasets import SELTODataset

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
    def __init__(self, voxel_grids, problem_information_list = None, num_query_points=1000, noise_std=0.05,
                 fixed_surface_points_size=2000, device='cpu', surface_sample_ratio=0.7,
                 anchor='min', voxel_dims=None, dataset = None):
        self.voxel_grids = voxel_grids
        self.num_query_points = num_query_points
        self.noise_std = noise_std
        self.fixed_surface_points_size = fixed_surface_points_size
        self.device = device
        self.surface_sample_ratio = surface_sample_ratio
        self.anchor = anchor  # 'min' | 'center' | 'max' : which corner to snap to in [-1,1]^3

        self.problem_information_list = problem_information_list
        self.dataset = dataset

        self.surface_data = []
        self.sdf_grids = []

        # For SELTO conditioning
        self.cond_grids = None    # will become a list of [C_cond, D, H, W] tensors
        self.F_scale = None       # global normalization for |F|

        # Compute shared normalization constants
        # Determine voxel dimensions: use explicit override if provided, else infer
        # from first voxel grid if available, else fall back to common default (39,39,21).
        if voxel_dims is not None:
            D, H, W = voxel_dims
        else:
            try:
                D, H, W = voxel_grids[0].shape
            except Exception:
                D, H, W = (39, 39, 21)
        # store voxel dims as float tensor for computations
        self.voxel_dims = torch.tensor([D, H, W], dtype=torch.float32)
        self.center = (self.voxel_dims - 1) / 2.0
        # coordinate scale: map longest axis to [-1,1]
        self.coord_scale = float(self.voxel_dims.max().item() / 2.0)
        # sdf scale: maximum possible distance from center to a corner (so normalized SDF <= 1)
        half = (self.voxel_dims - 1) / 2.0
        self.sdf_scale = float(torch.norm(half).item())

        for vg in voxel_grids:
            surface_points, normals = self._precompute_surface(vg)
            self.surface_data.append((surface_points, normals))
            self.sdf_grids.append(self._get_signed_distance_grid(vg.numpy()))

        if self.problem_information_list is not None:
            assert len(self.problem_information_list) == len(self.voxel_grids), "Length of problem_information_list must match number of voxel grids"
            if self.dataset is not None:
                self.dataset = 'SELTO'
            if self.dataset == 'SELTO':
                                # ------------------------------------------------------------
                # Assumes F_list and Ω_design_list are available in scope and
                # each has length == len(self.voxel_grids).
                # F_list:      list of arrays/tensors [3, D, H, W]
                # Ω_design_list: list of arrays/tensors [1, D, H, W] or [D, H, W]
                # ------------------------------------------------------------
                F_list = problem_information_list[0]
                Ω_design_list = problem_information_list[1]
                
                global_F_tensors = []
                for F in F_list:
                    F_t = torch.as_tensor(F, dtype=torch.float32)
                    assert F_t.shape[0] == 3, f"Expected F to have 3 channels, got {F_t.shape}"
                    # sanity check spatial dims if you want:
                    assert F_t.shape[1:] == (D, H, W), \
                        f"F spatial shape {F_t.shape[1:]} does not match voxel_dims {(D, H, W)}"
                    global_F_tensors.append(F_t)

                # Global normalization for F magnitude (avoid division by zero)
                max_F_val = 0.0
                for F_t in global_F_tensors:
                    if F_t.numel() > 0:
                        max_F_val = max(max_F_val, float(F_t.abs().max().item()))
                self.F_scale = max(max_F_val, 1e-8)

                self.cond_grids = []
                eps = 1e-8

                for F_t, Omega_np in zip(global_F_tensors, Ω_design_list):
                    # Convert Ω_design to tensor, ensure shape [1, D, H, W]
                    Omega_t = torch.as_tensor(Omega_np)
                    if Omega_t.ndim == 3:
                        Omega_t = Omega_t.unsqueeze(0)  # [1, D, H, W]
                    Omega_t = Omega_t.to(torch.int16)
                    assert Omega_t.shape[1:] == (D, H, W), \
                        f"Ω_design spatial shape {Omega_t.shape[1:]} does not match voxel_dims {(D, H, W)}"

                    # --- Process F: direction, magnitude, mask ---
                    # |F| per voxel: [1, D, H, W]
                    F_mag = torch.linalg.norm(F_t, dim=0, keepdim=True)  # sqrt(sum_c F_c^2)

                    # Direction field: [3, D, H, W]
                    F_dir = F_t / (F_mag + eps)

                    # Normalized magnitude: [1, D, H, W]
                    F_mag_norm = F_mag / self.F_scale

                    # Load mask: where any non-zero load is applied
                    load_mask = (F_mag > 0).to(torch.float32)  # [1, D, H, W]

                    # --- Process Ω_design: one-hot masks ---
                    # Ω_design ∈ {-1, 0, 1} in SELTO: free / void / solid
                    design_solid = (Omega_t == 1).to(torch.float32)   # [1, D, H, W]
                    design_void  = (Omega_t == 0).to(torch.float32)   # [1, D, H, W]
                    design_free  = (Omega_t == -1).to(torch.float32)  # [1, D, H, W]

                    # Concatenate into a conditioning grid:
                    # Channels: [F_dir(3), F_mag_norm(1), load_mask(1),
                    #           design_solid(1), design_void(1), design_free(1)]
                    cond = torch.cat(
                        [
                            F_dir,          # 3
                            F_mag_norm,     # 1
                            load_mask,      # 1
                            design_solid,   # 1
                            design_void,    # 1
                            design_free,    # 1
                        ],
                        dim=0
                    )  # -> [C_cond=8, D, H, W]

                    self.cond_grids.append(cond)

                # Optionally, you could stack into a big tensor:
                # self.cond_grids = torch.stack(self.cond_grids, dim=0)
                # shape would be [N, C_cond, D, H, W]

    def _precompute_surface(self, voxel_grid):
        voxel_np = voxel_grid.numpy()
        try:
            verts, faces, _, _ = measure.marching_cubes(voxel_np, level=0.5)
        except ValueError:
            return torch.empty((0, 3)), torch.empty((0, 3))

        # Defensive: ensure marching_cubes vertex axis-order matches voxel indexing
        verts_np = verts.copy()
        verts_np = self._permute_verts_if_needed(verts_np)

        # Normalize verts into [-1, 1]^3 (then optionally shift to anchor to a corner)
        verts = torch.tensor(verts_np, dtype=torch.float32)
        verts_normalized = (verts - self.center) / self.coord_scale

        # Anchor normalized coordinates to a chosen corner if requested.
        # 'min' -> snap the minimum voxel index (0) to -1 along each axis
        # 'max' -> snap the maximum voxel index (dims-1) to +1 along each axis
        # 'center' -> keep centered (default behavior except when anchor parameter set)
        if getattr(self, 'anchor', 'center') != 'center':
            if self.anchor == 'min':
                desired = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32)
                current = (-self.center) / self.coord_scale
            elif self.anchor == 'max':
                desired = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
                current = (self.voxel_dims - 1 - self.center) / self.coord_scale
            else:
                # unknown anchor string -> no shift
                desired = None
                current = None

            if desired is not None:
                shift = desired - current
                verts_normalized = verts_normalized + shift

        mesh = trimesh.Trimesh(vertices=verts_normalized.numpy(), faces=faces, process=False)
        # attempt basic repair to reduce degenerate faces/normals
        try:
            mesh = repair_mesh(mesh)
        except Exception:
            pass

        N = self.fixed_surface_points_size * 2
        try:
            surface_points, face_indices = mesh.sample(N, return_index=True)
            normals = mesh.face_normals[face_indices]
            return torch.tensor(surface_points, dtype=torch.float32), torch.tensor(normals, dtype=torch.float32)
        except Exception as e:
            print(f"[VoxelSDFDataset] mesh.sample failed: {e}")
            return torch.empty((0, 3)), torch.empty((0, 3))

    def _permute_verts_if_needed(self, verts_np):
        """Heuristic to detect and fix axis-order mismatches between marching_cubes verts
        and the voxel array indexing (voxel_dims = [D,H,W]).

        If a permutation of columns makes the vertex maxima reasonable relative to
        voxel dimensions, apply it and return the permuted array. Otherwise return
        the input unchanged.
        """
        try:
            vd = self.voxel_dims.cpu().numpy()
        except Exception:
            vd = np.array(self.voxel_dims)

        # quick guards
        if verts_np.size == 0:
            return verts_np

        vm = verts_np.max(axis=0)
        # if maxima are much larger than voxel dims, try permutations
        if np.any(vm > vd * 1.5):
            perms = [(0,1,2), (2,1,0), (1,0,2), (0,2,1), (2,0,1), (1,2,0)]
            for p in perms:
                vm_p = verts_np[:, p].max(axis=0)
                if np.all(vm_p <= vd * 1.5):
                    if p != (0,1,2):
                        print(f"[VoxelSDFDataset] Permuting marching_cubes verts axes with permutation {p}")
                    return verts_np[:, p]
        return verts_np

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

        # Normalize SDF by geometric scale so values are roughly in [-1,1]
        sdf_values = sdf_values / self.sdf_scale

        if self.cond_grids is not None:
            cond = self.cond_grids[idx].to(self.device)
            return surface_points.to(self.device), query_points, sdf_values.to(self.device), cond.to(self.device)
        else:
            return surface_points.to(self.device), query_points, sdf_values.to(self.device)

    def _process_to_fixed_size(self, points, normals):
        n_points = points.shape[0]
        if n_points == self.fixed_surface_points_size:
            return points, normals
        if n_points == 0:
            # return zeros if no surface points were found
            pts = torch.zeros(self.fixed_surface_points_size, 3, dtype=points.dtype)
            nrm = torch.zeros(self.fixed_surface_points_size, 3, dtype=normals.dtype)
            return pts, nrm
        if n_points < self.fixed_surface_points_size:
            # repeat points to reach desired size
            repeat = (self.fixed_surface_points_size + n_points - 1) // n_points
            pts = points.repeat(repeat, 1)
            nrms = normals.repeat(repeat, 1)
            indices = torch.randperm(pts.shape[0])[:self.fixed_surface_points_size]
            return pts[indices], nrms[indices]
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
    # If conditioning grids are present, items will have 4 elements
    if len(batch[0]) == 4:
        conds = torch.stack([item[3] for item in batch])  # [B, C_cond, D, H, W]
        return point_clouds, query_points, sdf_values, conds
    return point_clouds, query_points, sdf_values

def create_problem_information_lists(selto):
    Ω_design_list = []
    F_list = []
    for i in range(len(selto)):
        problem, solution = selto[i]
        Ω_design_list.append(problem.Ω_design)
        F_list.append(problem.F)
    return F_list, Ω_design_list