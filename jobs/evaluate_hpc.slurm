#!/bin/bash
#SBATCH --job-name=wikiart-evaluate
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/slurm_eval_%j.out
#SBATCH --error=outputs/logs/slurm_eval_%j.err

# Usage:
#   sbatch --account=f202500002hpcvlabistulg --partition=dev-a100-40 jobs/evaluate_hpc.slurm configs/config_mobilenetv3.yaml 123456__abcd1234
#   sbatch --account=f202500002hpcvlabistulg --partition=dev-a100-40 jobs/evaluate_hpc.slurm configs/config_efficientnetb3.yaml outputs/checkpoints/efficientnetb3/123456__abcd1234/best_model.keras

CONFIG=${1:-configs/config_efficientnetb3.yaml}
SELECTOR=${2:-}

cd "$SLURM_SUBMIT_DIR"

mkdir -p outputs/logs

echo "================================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Config     : $CONFIG"
if [[ -n "$SELECTOR" ]]; then
  echo "Selector   : $SELECTOR"
else
  echo "Selector   : <auto>"
fi
echo "Start time : $(date)"
echo "================================================"

# --- Debug info ---
echo "Python     : $(which python3) — $(python3 --version)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "test       : $(find data/test -name '*.jpg' 2>/dev/null | wc -l) images"
echo "------------------------------------------------"

module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load Python/3.11.3-GCCcore-12.3.0
source .venv/bin/activate

export XLA_FLAGS="--xla_gpu_cuda_data_dir=$EBROOTCUDA"

ARGS=(--config "$CONFIG")

if [[ -n "$SELECTOR" ]]; then
  if [[ "$SELECTOR" == *.keras || "$SELECTOR" == *.h5 ]]; then
    ARGS+=(--weights "$SELECTOR")
  else
    ARGS+=(--run-id "$SELECTOR")
  fi
fi

python3 src/evaluate.py "${ARGS[@]}"
EXIT_CODE=$?

echo "================================================"
echo "Exit code  : $EXIT_CODE"
echo "End time   : $(date)"
echo "================================================"

exit $EXIT_CODE
