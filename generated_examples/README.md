# Generated Examples

This directory is the default output location for longer checkpoint runs.

Each run writes into `generated_examples/<run-name>/`:

```text
run_config.json
loss_history.csv
loss_history.json
loss_plot.svg
train.log
sample.log
prior_sample.obj
```

The generated files are intentionally ignored by Git. Keep only this README and
the local ignore file in source control.
