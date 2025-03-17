import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from dl4to.datasets import SELTODataset


def generate_query_points(surface_points, num_query_points=1000, noise_std=0.05):
    """
    Generate query points around the surface points.
    
    Args:
        surface_points (torch.Tensor): Tensor of shape (N, 3) containing the (x, y, z) coordinates of surface points.
        num_query_points (int): Number of query points to generate.
        noise_std (float): Standard deviation of Gaussian noise added to the query points.
    
    Returns:
        query_points (torch.Tensor): Tensor of shape (num_query_points, 3) containing the (x, y, z) coordinates of query points.
    """
    # Randomly sample surface points
    sampled_indices = torch.randint(0, surface_points.shape[0], (num_query_points,))
    sampled_points = surface_points[sampled_indices]
    
    # Add Gaussian noise to the sampled points
    query_points = sampled_points + torch.randn_like(sampled_points) * noise_std
    
    return query_points

def voxel_to_sdf_data(voxel_grid, num_query_points=1000, noise_std=0.05):
    """
    Convert a binary voxel grid into surface points, normals, and query points suitable for SDF learning.
    
    Args:
        voxel_grid (torch.Tensor): Binary voxel grid of shape (D, H, W), where 1 indicates occupied and 0 indicates unoccupied.
        num_query_points (int): Number of query points to generate.
        noise_std (float): Standard deviation of Gaussian noise added to the query points.
    
    Returns:
        surface_points (torch.Tensor): Tensor of shape (N, 3) containing the (x, y, z) coordinates of surface points.
        normals (torch.Tensor): Tensor of shape (N, 3) containing the normal vectors at each surface point.
        query_points (torch.Tensor): Tensor of shape (num_query_points, 3) containing the (x, y, z) coordinates of query points.
    """
    surface_points, normals = extract_surface_points(voxel_grid)
    query_points = generate_query_points(surface_points, num_query_points, noise_std)
    
    return surface_points, normals, query_points

class VoxelSDFDataset(Dataset):
    def __init__(self, voxel_grids, num_query_points=1000, noise_std=0.05, fixed_surface_points_size=1000, device=None):
        """
        Custom Dataset for converting voxel grids into surface points, normals, and query points for SDF learning.
        
        Args:
            voxel_grids (list of torch.Tensor): List of binary voxel grids, each of shape (D, H, W).
            num_query_points (int): Number of query points to generate per voxel grid.
            noise_std (float): Standard deviation of Gaussian noise added to the query points.
            fixed_surface_points_size (int): Fixed size of the surface points tensor.
        """
        self.voxel_grids = voxel_grids
        self.num_query_points = num_query_points
        self.noise_std = noise_std
        self.fixed_surface_points_size = fixed_surface_points_size
        self.device = device

    def __len__(self):
        """Return the number of voxel grids in the dataset."""
        return len(self.voxel_grids)

    def __getitem__(self, idx):
        """
        Convert a voxel grid into surface points, normals, and query points.
        
        Args:
            idx (int): Index of the voxel grid to process.
        
        Returns:
            point_cloud (torch.Tensor): Tensor of shape (fixed_surface_points_size, 3) containing the (x, y, z) coordinates of surface points.
            normals (torch.Tensor): Tensor of shape (fixed_surface_points_size, 3) containing the normal vectors at each surface point.
            query_points (torch.Tensor): Tensor of shape (num_query_points, 3) containing the (x, y, z) coordinates of query points.
        """
        voxel_grid = self.voxel_grids[idx]
        
        # Extract surface points and normals
        surface_points, normals = self.extract_surface_points(voxel_grid)
        
        # Ensure surface_points and normals have a fixed size
        surface_points = self.fix_surface_points_size(surface_points)
        normals = self.fix_surface_points_size(normals)
        
        # Generate query points
        query_points = self.generate_query_points(surface_points, self.num_query_points, self.noise_std)
        
        return surface_points.to(self.device),  query_points.to(self.device), normals.to(self.device),

    @staticmethod
    def extract_surface_points(voxel_grid):
        """
        Extract surface points and normals from a binary voxel grid.
        Only points strictly on the surface are included (no interior points).

        Args:
            voxel_grid (torch.Tensor): Binary voxel grid of shape (D, H, W), where 1 indicates occupied and 0 indicates unoccupied.

        Returns:
            surface_points (torch.Tensor): Tensor of shape (N, 3) containing the (x, y, z) coordinates of surface points.
            normals (torch.Tensor): Tensor of shape (N, 3) containing the normal vectors at each surface point.
        """
        # Pad the voxel grid to handle edge cases
        padded_grid = F.pad(voxel_grid.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), mode='constant', value=0).squeeze()

        # Original dimensions
        D, H, W = voxel_grid.shape

        # Create a mask for surface voxels
        surface_mask = torch.zeros_like(voxel_grid, dtype=torch.bool)

        # Check all 6 neighbors for each voxel
        for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
            # Shift the grid and compare with the original
            shifted_grid = padded_grid[1 + dx:D + 1 + dx, 1 + dy:H + 1 + dy, 1 + dz:W + 1 + dz]
            surface_mask |= (voxel_grid == 1) & (shifted_grid == 0)

        # Extract coordinates of surface voxels
        surface_indices = torch.nonzero(surface_mask, as_tuple=False).float()

        # Normalize to [-1, 1]
        surface_points = (surface_indices / torch.tensor([D, H, W]).float()) * 2 - 1

        # Compute normals using central differences
        gradient_x = padded_grid[1:D+1, 1:H+1, 2:W+2] - padded_grid[1:D+1, 1:H+1, :-2]  # Shape: (D, H, W-1)
        gradient_y = padded_grid[1:D+1, 2:H+2, 1:W+1] - padded_grid[1:D+1, :-2, 1:W+1]  # Shape: (D, H-1, W)
        gradient_z = padded_grid[2:D+2, 1:H+1, 1:W+1] - padded_grid[:-2, 1:H+1, 1:W+1]  # Shape: (D-1, H, W)

        # Truncate to a common shape
        min_D = min(gradient_x.shape[0], gradient_y.shape[0], gradient_z.shape[0])
        min_H = min(gradient_x.shape[1], gradient_y.shape[1], gradient_z.shape[1])
        min_W = min(gradient_x.shape[2], gradient_y.shape[2], gradient_z.shape[2])

        gradient_x = gradient_x[:min_D, :min_H, :min_W]
        gradient_y = gradient_y[:min_D, :min_H, :min_W]
        gradient_z = gradient_z[:min_D, :min_H, :min_W]

        # Compute normals from the gradient
        normals = torch.stack([gradient_x[surface_mask[:min_D, :min_H, :min_W]],
                            gradient_y[surface_mask[:min_D, :min_H, :min_W]],
                            gradient_z[surface_mask[:min_D, :min_H, :min_W]]], dim=-1)
        normals = F.normalize(normals, p=2, dim=-1)  # Normalize to unit vectors

        return surface_points, normals

    def fix_surface_points_size(self, surface_points):
        """
        Ensure that the surface points tensor has a fixed size by either padding or sampling.
        
        Args:
            surface_points (torch.Tensor): Tensor of shape (N, 3) containing the (x, y, z) coordinates of surface points.
        
        Returns:
            surface_points (torch.Tensor): Tensor of shape (fixed_surface_points_size, 3) containing the (x, y, z) coordinates of surface points.
        """
        num_points = surface_points.shape[0]
        
        if num_points < self.fixed_surface_points_size:
            # Pad with zeros if there are not enough points
            padding = torch.zeros(self.fixed_surface_points_size - num_points, 3)
            surface_points = torch.cat([surface_points, padding], dim=0)
        elif num_points > self.fixed_surface_points_size:
            # Randomly sample points if there are too many points
            indices = torch.randperm(num_points)[:self.fixed_surface_points_size]
            surface_points = surface_points[indices]
        
        return surface_points

    def generate_query_points(self, surface_points, num_query_points, noise_std):
        """
        Generate query points around the surface points with added noise.
        
        Args:
            surface_points (torch.Tensor): Tensor of shape (N, 3) containing the (x, y, z) coordinates of surface points.
            num_query_points (int): Number of query points to generate.
            noise_std (float): Standard deviation of Gaussian noise added to the query points.
        
        Returns:
            query_points (torch.Tensor): Tensor of shape (num_query_points, 3) containing the (x, y, z) coordinates of query points.
        """
        # Generate query points by adding noise to the surface points
        query_points = surface_points + torch.rand_like(surface_points) * noise_std
        
        return query_points

def create_voxel_grids(dataset):
    voxel_grids = []
    for model_idx in range(len(dataset)):
        # Retrieve the model data
        problem, solution = dataset[model_idx]
        density = solution.θ.squeeze().cpu()  # Density or SDF tensor
        voxel_grids.append(density)
    return voxel_grids

def prepare_data_for_model(voxel_grid, num_query_points=1000, noise_std=0.05):
    """
    Prepare data for the Diffusion-SDF model from a binary voxel grid.
    
    Args:
        voxel_grid (torch.Tensor): Binary voxel grid of shape (D, H, W), where 1 indicates occupied and 0 indicates unoccupied.
        num_query_points (int): Number of query points to generate.
        noise_std (float): Standard deviation of Gaussian noise added to the query points.
    
    Returns:
        point_cloud (torch.Tensor): Tensor of shape (1, N, 3) containing the (x, y, z) coordinates of surface points.
        normals (torch.Tensor): Tensor of shape (1, N, 3) containing the normal vectors at each surface point.
        query_points (torch.Tensor): Tensor of shape (1, num_query_points, 3) containing the (x, y, z) coordinates of query points.
    """
    surface_points, normals, query_points = voxel_to_sdf_data(voxel_grid, num_query_points, noise_std)
    
    # Add batch dimension
    point_cloud = surface_points.unsqueeze(0)  # Shape: (1, N, 3)
    normals = normals.unsqueeze(0)            # Shape: (1, N, 3)
    query_points = query_points.unsqueeze(0)  # Shape: (1, num_query_points, 3)
    
    return point_cloud, normals, query_points
    
def collate_fn(batch):
    """
    Collate function for the DataLoader to handle variable-sized surface points.
    
    Args:
        batch (list of tuples): List of (surface_points, query_points) tuples.
    
    Returns:
        point_clouds (list of torch.Tensor): List of surface points tensors, each of shape (N_i, 3).
        query_points (torch.Tensor): Stacked query points tensor of shape (batch_size, num_query_points, 3).
    """
    point_clouds = [item[0] for item in batch]
    query_points = torch.stack([item[1] for item in batch])
    normals = torch.stack([item[2] for item in batch])
    return point_clouds, query_points, normals

# print("Loading SELTO dataset...")
# selto = SELTODataset(root='.', name='sphere_complex', train=True)
# print("SELTO dataset loaded!")
# print("Constructing voxel grids...")
# breakpoint()
# voxel_grids = create_voxel_grids(selto)
# voxel_grid = voxel_grids[0]
# breakpoint()
# # Extract surface points
# surface_points, normals = VoxelSDFDataset.extract_surface_points(voxel_grid)

# # Save the extracted surface as an OBJ file
# def save_to_obj(filename, vertices, normals=None):
#     with open(filename, 'w') as f:
#         for i, v in enumerate(vertices):
#             f.write(f"v {v[0]} {v[1]} {v[2]}\n")  # Write vertex
#             if normals is not None:
#                 f.write(f"vn {normals[i][0]} {normals[i][1]} {normals[i][2]}\n")  # Write normal

# # Save the extracted points
# save_to_obj("surface.obj", surface_points.numpy(), normals.numpy())

# print("OBJ file saved! Open 'surface.obj' in a 3D viewer.")