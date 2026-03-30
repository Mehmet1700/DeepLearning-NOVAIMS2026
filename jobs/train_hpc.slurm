#!/bin/bash
#SBATCH --job-name=wikiart-train
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Usage:
#   sbatch jobs/train_hpc.slurm configs/config_resnet50.yaml
#   sbatch jobs/train_hpc.slurm configs/config_efficientnetb3.yaml
#   sbatch jobs/train_hpc.slurm configs/config_vgg16.yaml

CONFIG=${1:-configs/config_efficientnetb3.yaml}

echo "================================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Config     : $CONFIG"
echo "Start time : $(date)"
echo "================================================"

# Load modules (adjust to your HPC environment)
module load python/3.10
module load cuda/11.8

# Activate virtual environment
source .venv/bin/activate

# Create output dirs if they don't exist
mkdir -p outputs/checkpoints outputs/logs

# Run training
python src/train.py --config $CONFIG

echo "================================================"
echo "End time : $(date)"
echo "================================================"
