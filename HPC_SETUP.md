# Running the Project on Deucalion HPC

This guide explains how to access and run the WikiArt Deep Learning project on the Deucalion HPC cluster.

---

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

Optional: Add to your `~/.ssh/config` for easier login:

```
Host deucalion
    HostName login.deucalion.macc.fccn.pt
    User mehmet1700
    IdentityFile ~/.ssh/id_ed25519_deucalion
```

Then simply: `ssh deucalion`

---

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

---

## 3. Submitting Training Jobs

### Standard command

Account and partition must be passed via the command line — do **not** put them in the SLURM script header, as Deucalion ignores or rejects them there.

```bash
sbatch --account=f202500002hpcvlabistulg \
       --partition=dev-a100-40 \
       jobs/train_hpc.slurm configs/config_efficientnetb3.yaml
```

You can submit multiple models in parallel:

```bash
sbatch --account=f202500002hpcvlabistulg --partition=dev-a100-40 jobs/train_hpc.slurm configs/config_resnet50.yaml
sbatch --account=f202500002hpcvlabistulg --partition=dev-a100-40 jobs/train_hpc.slurm configs/config_efficientnetb3.yaml
sbatch --account=f202500002hpcvlabistulg --partition=dev-a100-40 jobs/train_hpc.slurm configs/config_vgg16.yaml
```

### Available partitions

| Partition | GPU | Max Time | Use for |
|---|---|---|---|
| `dev-a100-40` | A100 40GB | 4 hours | Training ⭐ Recommended |
| `dev-a100-80` | A100 80GB | 4 hours | Large models / big batches |
| `dev-x86` | CPU only | 4 hours | Testing / debugging |
| `normal-a100-40` | A100 40GB | 2 days | Long training runs |
| `normal-a100-80` | A100 80GB | 2 days | Long training runs |

> ⚠️ **Max runtime per job is 4 hours** on `dev-*` partitions. Use `normal-*` for runs longer than 4 hours.

### Account

| Account | Partition |
|---|---|
| `f202500002hpcvlabistulg` | All GPU partitions |

---

## 4. Monitoring Jobs

```bash
# See your running/queued jobs
squeue -u $USER

# Watch live output log
cat outputs/logs/slurm_<job_id>.out

# Cancel a job
scancel <job_id>

# See completed jobs
sacct -u $USER

# Get detailed info about a specific job
sacct -j <job_id> -l
```

---

## 5. GPU Verification

```bash
# Check available GPUs on your node
nvidia-smi

# Monitor GPU usage in real-time
nvidia-smi -l 1
```

---

## 6. Useful Commands

```bash
# Check disk usage in the project directory
du -sh /projects/F202500002HPCVLABISTUL/Mehmet1700/DeepLearningProject-NOVAIMS2026

# Check available partitions and GPU types
sinfo -o "%P %G %l %D %N"

# Check your account associations
sacctmgr show associations user=$USER format=account%50 --noheader
```
