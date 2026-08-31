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

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES_PER_DATASET="${MAX_SAMPLES_PER_DATASET:-256}"
NUM_QUERY_POINTS="${NUM_QUERY_POINTS:-5000}"
FIXED_SURFACE_POINTS_SIZE="${FIXED_SURFACE_POINTS_SIZE:-10000}"
NOISE_STD="${NOISE_STD:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
BETA_KL="${BETA_KL:-1e-5}"
PRIOR_STD="${PRIOR_STD:-0.25}"
SAVE_EVERY="${SAVE_EVERY:-5}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints_mod}"
VAE_CHECKPOINT_ROOT="${VAE_CHECKPOINT_ROOT:-checkpoints_vae}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-generated_examples}"
SAMPLE_AFTER_TRAIN="${SAMPLE_AFTER_TRAIN:-1}"
SAMPLE_GRID="${SAMPLE_GRID:-32}"
SAMPLE_CHUNK_SIZE="${SAMPLE_CHUNK_SIZE:-50000}"

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
  artifact_dir="${ARTIFACT_ROOT}/${run_name}"
  mkdir -p "${artifact_dir}"
  echo "=== Training ${run_name} from dataset '${dataset}' ==="
  python trainer.py \
    --dataset-name "${dataset}" \
    --run-name "${run_name}" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --vae-checkpoint-root "${VAE_CHECKPOINT_ROOT}" \
    --num-epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --max-samples-per-dataset "${MAX_SAMPLES_PER_DATASET}" \
    --num-query-points "${NUM_QUERY_POINTS}" \
    --fixed-surface-points-size "${FIXED_SURFACE_POINTS_SIZE}" \
    --noise-std "${NOISE_STD}" \
    --learning-rate "${LEARNING_RATE}" \
    --beta-kl "${BETA_KL}" \
    --prior-std "${PRIOR_STD}" \
    --save-every "${SAVE_EVERY}" \
    --artifact-dir "${artifact_dir}" 2>&1 | tee "${artifact_dir}/train.log"

  if [[ "${SAMPLE_AFTER_TRAIN}" == "1" ]]; then
    sample_dataset="${dataset}"
    if [[ "${dataset}" == "all" ]]; then
      sample_dataset="sphere_complex"
    fi
    echo "=== Sampling ${run_name} prior example ==="
    if python sample_sdf_obj.py \
      --ckpt "${CHECKPOINT_ROOT}/${run_name}/mod_last.pth" \
      --dataset-name "${sample_dataset}" \
      --grid "${SAMPLE_GRID}" \
      --chunk-size "${SAMPLE_CHUNK_SIZE}" \
      --prior-sigma "${PRIOR_STD}" \
      --pad-boundary \
      --repair-mesh \
      --outfile "${artifact_dir}/prior_sample.obj" > "${artifact_dir}/sample.log" 2>&1; then
      echo "Sample written to ${artifact_dir}/prior_sample.obj"
    else
      echo "Sample generation failed for ${run_name}; see ${artifact_dir}/sample.log"
    fi
  fi
done

echo "=== Checkpoints written under ${CHECKPOINT_ROOT}/* and ${VAE_CHECKPOINT_ROOT}/* ==="
echo "=== Run artifacts written under ${ARTIFACT_ROOT}/* ==="
