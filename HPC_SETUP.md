# Running the Project on Deucalion HPC

This guide explains how to access and run the WikiArt Deep Learning project on the Deucalion HPC cluster.

## 1. SSH Access

**First-time setup:**

Ask Mehmet for access credentials and to the cluster. Each person should get their own ssh key. 

On your local machine:

```bash
# Generate your own SSH key (if you don't have one)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_deucalion

# Copy YOUR public key to the HPC cluster
ssh-copy-id -i ~/.ssh/id_ed25519_deucalion.pub mehmet1700@login.deucalion.macc.fccn.pt

# Now you can SSH without a password
ssh mehmet1700@login.deucalion.macc.fccn.pt
```

**Why not share keys?**
- **No accountability** — Can't track who ran what
- **Security risk** — If compromised, everyone's affected
- **No revocation** — Can't remove just one person's access
- **Auditing nightmare** — Logs won't show who did what

Optional: Add to your `~/.ssh/config` for easier login:

```
Host deucalion
    HostName login.deucalion.macc.fccn.pt
    User mehmet1700
    IdentityFile ~/.ssh/id_ed25519_deucalion
```

Then simply: `ssh deucalion`

## 2. Project Setup

Once connected to the cluster:

```bash
# Navigate to the shared project
cd /projects/F202500002HPCVLABISTUL/Mehmet1700/DeepLearningProject-NOVAIMS2026

# Create a Python virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 3. Running the Notebook

### Option A: Convert to Python Script (Simplest)

```bash
# Convert the notebook to a Python script
jupyter nbconvert --to script src/explore_wikiart.ipynb --output ../explore_wikiart_run.py

# Run it
python explore_wikiart_run.py
```

This will process the images and print results to the terminal.

### Option B: Batch Job (For Long Runs)

For heavy processing, submit a batch job to the Slurm scheduler:

Create `run_notebook.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=wikiart-explore
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --output=logs/slurm_%j.log

source .venv/bin/activate
jupyter nbconvert --to html --execute src/explore_wikiart.ipynb --output ../results/explore_wikiart.html
```

Submit it:

```bash
sbatch run_notebook.sh
```

Check status:

```bash
squeue -u $USER
```

## 4. Useful Commands

### Check Available GPUs

```bash
# See all GPUs on the current node
nvidia-smi

# See GPU memory available
nvidia-smi --query-gpu=memory.free --format=csv

# Monitor GPU usage in real-time
nvidia-smi -l 1  # Updates every 1 second
```

### Check Cluster Info

```bash
# See all available nodes and their state
sinfo

# See available GPUs across the cluster
sinfo -N  # Lists nodes with details

# Check node specifications
sinfo -N -l  # Long format with more details
```

### Request Hardware Resources (Interactive Job)

For testing or interactive work:

```bash
# Request 1 GPU for interactive use (2 hours)
salloc --gpus=1 --mem=16G --time=02:00:00
```

### Available Resources

Choose the right account and partition for your needs:

| Architecture | Account | Partition | Best For |
|---|---|---|---|
| **x86 CPU** | `f202500002hpcvlabistulx` | `dev-x86` | CPU-only tasks, small jobs, testing |
| **ARM CPU** | `f202500002hpcvlabistula` | `dev-arm` | ARM-optimized workloads |
| **GPU (A100)** | `f202500002hpcvlabistulg` | `dev-a100-80` | Deep learning, GPU-intensive compute ⭐ Recommended for this project |

**GPU Time Limits & Quota:**
- ⏱️ **Maximum runtime per job:** 4 hours (240 minutes)
- 📊 **Project allocation:** 49,920,000 CPU-minutes (≈95 GPU-years, shared)
- 📌 For jobs longer than 4 hours, submit multiple consecutive jobs

**Example: Request GPU resources**

```bash
# Interactive job with GPU
srun --account=f202500002hpcvlabistulg \
     --partition=dev-a100-80 \
     --gres=gpu:1 \
     --mem=16G \
     --time=00:10:00 \
     --pty bash
```

**Example: Batch job with GPU** (add to your submit_job.sh):

```bash
#SBATCH --account=f202500002hpcvlabistulg
#SBATCH --partition=dev-a100-80
#SBATCH --gpus=1
```

### Submit Batch Jobs

For longer runs (already shown above, but here's a complete example):

```bash
# Create submit_job.sh with:
#!/bin/bash
#SBATCH --job-name=my-job
#SBATCH --gpus=1           # Request 1 GPU
#SBATCH --mem=16G          # Memory
#SBATCH --time=02:00:00    # Max 2 hours (HH:MM:SS)
#SBATCH --output=logs/job_%j.log
#SBATCH --error=logs/job_%j.err

source .venv/bin/activate
python my_script.py

# Submit it:
sbatch submit_job.sh
```

### Monitor Jobs

```bash
# See your running jobs
squeue -u $USER

# See job details with more info
squeue -u $USER -l

# Cancel a job
scancel <job_id>  # Get job_id from squeue

# See completed jobs and their status
sacct -u $USER  # Shows recent jobs

# Get detailed info about a specific job
sacct -j <job_id> -l
```

### Other Useful Commands

```bash
# Check disk usage in your home directory
du -sh ~

# Check disk usage in the project directory
du -sh /projects/F202500002HPCVLABISTUL/Mehmet1700/DeepLearningProject-NOVAIMS2026

# See available modules (software)
module avail

# Load a module (e.g., CUDA, Python, etc.)
module load cuda/11.8

# See currently loaded modules
module list
```
