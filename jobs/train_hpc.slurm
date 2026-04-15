#!/bin/bash
#SBATCH --job-name=wikiart-train
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err

# Usage:
#   Production training (up to 2 days, recommended):
#     sbatch --account=f202500002hpcvlabistulg --partition=normal-a100-40 jobs/train_hpc.slurm configs/config_mobilenetv3.yaml
#   Quick tests (4hr cap):
#     sbatch --account=f202500002hpcvlabistulg --partition=dev-a100-40 jobs/train_hpc.slurm configs/config_mobilenetv3.yaml

CONFIG=${1:-configs/config_efficientnetb3.yaml}

# Navigate to project directory
cd "$SLURM_SUBMIT_DIR"

mkdir -p outputs/checkpoints outputs/logs

echo "================================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Config     : $CONFIG"
echo "Start time : $(date)"
echo "================================================"

# --- Environment setup ---
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load Python/3.11.3-GCCcore-12.3.0
source .venv/bin/activate

# Point XLA to the CUDA nvvm/libdevice compiler (set by EasyBuild module)
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$EBROOTCUDA"
# Suppress TF info/warning spam in logs
export TF_CPP_MIN_LOG_LEVEL=2
# Dedicated GPU threads — reduces scheduling overhead on single-GPU jobs
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=1
# TF Hub cache — allows compute nodes (no internet) to load pre-downloaded models
export TFHUB_CACHE_DIR="$(pwd)/.tfhub_cache"

# --- Debug info ---
echo "Python     : $(which python3) — $(python3 --version)"
echo "Working dir: $(pwd)"
echo "Config     : $CONFIG"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "------------------------------------------------"

echo "Starting training..."
python3 src/train.py --config $CONFIG
EXIT_CODE=$?

echo "================================================"
echo "Exit code  : $EXIT_CODE"
echo "End time   : $(date)"
echo "================================================"

exit $EXIT_CODE