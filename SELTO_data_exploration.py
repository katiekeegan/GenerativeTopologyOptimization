from utils.preprocess_data import *
import dl4to
from dl4to.datasets import SELTODataset

selto = SELTODataset(root='.', name='disc_simple', train=True)
voxel_grids = create_voxel_grids(selto)
Ω_design_list = []
Ω_dirichlet_list = []
ν_list = []
σ_ys_list = []
F_list = []
E_list = []
for i in range(len(selto)):
    problem, solution = selto[i]
    Ω_design_list.append(problem.Ω_design)
    Ω_dirichlet_list.append(problem.Ω_dirichlet)
    ν_list.append(problem.ν)
    σ_ys_list.append(problem.σ_ys)
    F_list.append(problem.F)
    E_list.append(problem.E)

import torch

Ωd0 = Ω_dirichlet_list[0]
ΩD0 = Ω_design_list[0]
F0 = F_list[0]

all_dirichlet_equal = all(torch.equal(Ωd0, Ωd) for Ωd in Ω_dirichlet_list[1:])
all_design_equal     = all(torch.equal(ΩD0, ΩD) for ΩD in Ω_design_list[1:])
all_F_equal          = all(torch.equal(F0, F) for F in F_list[1:])
print("All Ω_dirichlet equal?", all_dirichlet_equal)
print("All Ω_design equal?", all_design_equal)
print("All F equal?", all_F_equal)
print(f'Number of unique ν: {len(set(ν_list))}')        
print(f'Number of unique σ_ys: {len(set(σ_ys_list))}')
print(f'Number of unique E: {len(set(E_list))}')
breakpoint()