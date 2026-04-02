#!/bin/bash
#SBATCH --job-name=wikiart-evaluate
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/projects/F202500002HPCVLABISTUL/Mehmet1700/DeepLearningProject-NOVAIMS2026/outputs/logs/slurm_eval_%j.out
#SBATCH --error=/projects/F202500002HPCVLABISTUL/Mehmet1700/DeepLearningProject-NOVAIMS2026/outputs/logs/slurm_eval_%j.err

# Usage:
#   sbatch --account=f202500002hpcvlabistulg --partition=dev-a100-40 jobs/evaluate_hpc.slurm configs/config_mobilenetv3.yaml outputs/checkpoints/mobilenetv3/best_model.keras
#   sbatch --account=f202500002hpcvlabistulg --partition=dev-a100-40 jobs/evaluate_hpc.slurm configs/config_efficientnetb3.yaml outputs/checkpoints/efficientnetb3/best_model.keras

CONFIG=${1:-configs/config_efficientnetb3.yaml}
WEIGHTS=${2:-outputs/checkpoints/efficientnetb3/best_model.keras}

cd /projects/F202500002HPCVLABISTUL/Mehmet1700/DeepLearningProject-NOVAIMS2026

mkdir -p outputs/logs

echo "================================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Config     : $CONFIG"
echo "Weights    : $WEIGHTS"
echo "Start time : $(date)"
echo "================================================"

# --- Debug info ---
echo "Python     : $(which python3) — $(python3 --version)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "test       : $(find data/test -name '*.jpg' 2>/dev/null | wc -l) images"
echo "------------------------------------------------"

module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
source .venv/bin/activate

python3 src/evaluate.py --config $CONFIG --weights $WEIGHTS
EXIT_CODE=$?

echo "================================================"
echo "Exit code  : $EXIT_CODE"
echo "End time   : $(date)"
echo "================================================"

exit $EXIT_CODE
