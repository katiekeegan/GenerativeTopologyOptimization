#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --constraint=gpuhbm80g
#SBATCH --gpus=1
#SBATCH --account=m5357
#SBATCH --job-name=topology-selto-ckpt
#SBATCH --output=logs/topology-selto-ckpt-%j.out
#SBATCH --error=logs/topology-selto-ckpt-%j.err

set -euo pipefail

cd /pscratch/sd/k/katiekee/GenerativeTopologyOptimization
mkdir -p logs

./scripts/train_selto_checkpoints.sh
