#!/usr/bin/env bash
set -euo pipefail

module load python
module load pytorch/2.6.0

if [[ -n "${TOPOLOGY_FORMAT_CONVERTER_PATH:-}" ]]; then
  export PYTHONPATH="${TOPOLOGY_FORMAT_CONVERTER_PATH}:${PYTHONPATH:-}"
fi
export PYTHONPATH="${PYTHONPATH:-}:dl4to:."

python - <<'PY'
import topology_format_converter
print(f"Using topology_format_converter from {topology_format_converter.__file__}")
PY

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES_PER_DATASET="${MAX_SAMPLES_PER_DATASET:-256}"
NUM_QUERY_POINTS="${NUM_QUERY_POINTS:-5000}"
FIXED_SURFACE_POINTS_SIZE="${FIXED_SURFACE_POINTS_SIZE:-10000}"
NOISE_STD="${NOISE_STD:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
BETA_KL="${BETA_KL:-1e-5}"
PRIOR_STD="${PRIOR_STD:-0.25}"
SAVE_EVERY="${SAVE_EVERY:-1}"

DATASETS=(
  "disc_simple"
  "disc_complex"
  "sphere_simple"
  "sphere_complex"
  "all"
)

RUN_NAMES=(
  "disc_simple"
  "disc_complex"
  "sphere_simple"
  "sphere_complex"
  "combined_all"
)

for idx in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$idx]}"
  run_name="${RUN_NAMES[$idx]}"
  echo "=== Training ${run_name} from dataset '${dataset}' ==="
  python trainer.py \
    --dataset-name "${dataset}" \
    --run-name "${run_name}" \
    --num-epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --max-samples-per-dataset "${MAX_SAMPLES_PER_DATASET}" \
    --num-query-points "${NUM_QUERY_POINTS}" \
    --fixed-surface-points-size "${FIXED_SURFACE_POINTS_SIZE}" \
    --noise-std "${NOISE_STD}" \
    --learning-rate "${LEARNING_RATE}" \
    --beta-kl "${BETA_KL}" \
    --prior-std "${PRIOR_STD}" \
    --save-every "${SAVE_EVERY}"
done

echo "=== Checkpoints written under checkpoints_mod/* and checkpoints_vae/* ==="
