# Generative Topology Optimization (Latent Diffusion + SDF)

Brief overview of the current workflow combining a VAE + SDF prediction network with a latent-space DDPM for generative shape sampling. Current workflow is entirely based on [Diffusion-SDF](https://arxiv.org/pdf/2211.13757) using the [SELTO](https://arxiv.org/pdf/2209.05098) dataset as provided through [DL4TO](https://github.com/dl4to/dl4to).

## Components

- `trainer.py`: Trains the modulation module (VAE + SDF network). The VAE learns a latent representation; the SDF network predicts signed distance values conditioned on the VAE decoder output.
- `trainer_diffusion.py`: Trains a DDPM (diffusion model) directly in the latent space learned by the VAE (noise prediction on z).
- `sample_sdf_obj.py`: Samples a latent vector directly from a Gaussian prior (without diffusion) and exports a mesh via SDF evaluation + marching cubes.
- `sample_diffusion_model.py`: Performs reverse diffusion sampling to obtain a latent vector, decodes it, evaluates the SDF on a dense grid, and exports a mesh.

## Training the VAE + SDF Network

```bash
python trainer.py --sdf_focus --sdf_sign_loss  # optional flags
```

Key flags (see script for full list):
- `--sdf_focus`: emphasize samples near the zero-level set.
- `--sdf_sign_loss`: hinge-style sign consistency near surface.

Outputs:
- Checkpoints in `checkpoints_mod/` (modulation module) and `checkpoints_vae/`.
- Final state dict optionally saved as `modulation_module.pth`.

## Training the Latent Diffusion Model

Run after you have a trained modulation (VAE+SDF) checkpoint:

```bash
python trainer_diffusion.py --ckpt checkpoints_mod/mod_last.pth --epochs 100 --batch-size 8
```

Important flags:
- `--timesteps`: number of diffusion steps (default 1000).
- `--latent-dim`: must match the latent dimension used in `trainer.py`.
- `--dataset-name`: dataset identifier (defaults to `sphere_complex`).

Outputs:
- Diffusion checkpoints in `checkpoints_diffusion/` (e.g. `diffusion_epoch_50.pth`).

## Sampling (Gaussian Prior vs Diffusion)

Direct prior sampling (no diffusion):
```bash
python sample_sdf_obj.py --ckpt checkpoints_mod/mod_last.pth --grid 64 --outfile prior_sample.obj
```

Diffusion sampling (reverse DDPM in latent space):
```bash
python sample_diffusion_model.py \
	--modulation-ckpt checkpoints_mod/mod_last.pth \
	--diffusion-ckpt checkpoints_diffusion/diffusion_epoch_50.pth \
	--grid 64 --outfile diffusion_sample.obj --pad-boundary --repair
```

Useful flags for `sample_diffusion_model.py`:
- `--pad-boundary`: conservative boundary padding to enclose surface.
- `--repair`: attempt hole filling and cleanup (trimesh + optional pymeshfix).

## Folder Notes

- `checkpoints_mod/`, `checkpoints_vae/`, `checkpoints_diffusion/`: model checkpoints.

## Troubleshooting

- Mesh not watertight: try `--pad-boundary` and higher `--grid` resolution (e.g. 96 or 128). Ensure diffusion and VAE latent dims match.
- Empty / invalid surface (no zero crossing): verify the modulation checkpoint quality or adjust conditioning (e.g. re-train with more epochs).
- CUDA memory issues: lower `--grid` or `--chunk-size` during sampling; reduce batch size during training.