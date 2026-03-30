#!/bin/bash
#SBATCH --job-name=wikiart-train
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/projects/F202500002HPCVLABISTUL/Mehmet1700/DeepLearningProject-NOVAIMS2026/outputs/logs/slurm_%j.out
#SBATCH --error=/projects/F202500002HPCVLABISTUL/Mehmet1700/DeepLearningProject-NOVAIMS2026/outputs/logs/slurm_%j.err

# Usage:
#   sbatch --account=f202500002hpcvlabistulg --partition=dev-a100-40 jobs/train_hpc.slurm configs/config_resnet50.yaml
#   sbatch --account=f202500002hpcvlabistulg --partition=dev-a100-40 jobs/train_hpc.slurm configs/config_efficientnetb3.yaml
#   sbatch --account=f202500002hpcvlabistulg --partition=dev-a100-40 jobs/train_hpc.slurm configs/config_vgg16.yaml

CONFIG=${1:-configs/config_efficientnetb3.yaml}

# Navigate to project directory
cd /projects/F202500002HPCVLABISTUL/Mehmet1700/DeepLearningProject-NOVAIMS2026

mkdir -p outputs/checkpoints outputs/logs

echo "================================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Config     : $CONFIG"
echo "Start time : $(date)"
echo "================================================"

# --- Environment setup ---
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
source .venv/bin/activate

# --- Debug info ---
echo "Python     : $(which python3) — $(python3 --version)"
echo "Working dir: $(pwd)"
echo "Config     : $CONFIG"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "train      : $(find data/train -name '*.jpg' 2>/dev/null | wc -l) images"
echo "validation : $(find data/validation -name '*.jpg' 2>/dev/null | wc -l) images"
echo "test       : $(find data/test -name '*.jpg' 2>/dev/null | wc -l) images"
echo "------------------------------------------------"

echo "Starting training..."
python3 src/train.py --config $CONFIG
EXIT_CODE=$?

echo "================================================"
echo "Exit code  : $EXIT_CODE"
echo "End time   : $(date)"
echo "================================================"

exit $EXIT_CODE
