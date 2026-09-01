# Generated Examples

This directory is the default output location for longer checkpoint runs.
Generated files are run artifacts: loss plots, random prior samples, and
dataset-sample reconstructions.

Each run writes into `generated_examples/<run-name>/`:

```text
run_config.json
loss_history.csv
loss_history.json
loss_plot.svg
train.log
sample.log
prior_sample.obj
reconstruction_sample0_sdfs.npz
reconstruction_sample0_gt.obj
reconstruction_sample0_pred.obj
```

`prior_sample.obj` is a random VAE-prior draw and is the only artifact here
intended to represent an unconditioned model sample. It can fail early in
training if the predicted SDF does not cross zero. The reconstruction files use
sample 0 from the relevant dataset and include both the ground-truth OBJ and
the model prediction when marching cubes succeeds.

The generated files are intentionally ignored by Git. Keep only this README and
the local ignore file in source control.
