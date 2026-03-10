---
name: ai-cloud
description: "Use when: working with AAU AI Cloud HPC cluster. Covers Singularity container building, job submission, resource quotas (normal/unprivileged/deadline), and job monitoring via Slurm. Commands for building, running, and inspecting GPU/CPU workloads."
---
# AI Cloud HPC Skill

**Use when**: Building Singularity containers, submitting jobs to AI Cloud, monitoring job queues, managing Slurm parameters, requesting beyond-default resources.

**Scope**: AAU’s AI Cloud cluster ([https://hpc.aau.dk/ai-cloud/](https://hpc.aau.dk/ai-cloud/)). Singularity 3.5+, Slurm scheduler, GPU/CPU job management.

---

## Quick Reference

Task

Command

Notes

View job queue

`squeue`

All jobs; use `squeue --me` for your own

Submit job

`sbatch script.sh`

Via batch script

Run interactive

`srun command`

Direct execution with default QOS

Build container

`sbatch build.sh`

Requires Singularity def file

Cancel job

`scancel JOBID`

Remove by job ID

Check GPU util

`nvidia-smi`

GPU resource usage (on compute node)

---

## 1\. Building Singularity Containers

### Workflow

1.  **Create definition file** (`.def`) with base image + software
2.  **Create batch build script** to submit to scheduler
3.  **Submit build** with `sbatch`
4.  **Test container** with `srun`

### Step 1: Create Definition File

```singularity
Bootstrap: docker
From: pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel

%environment
    export LC_ALL=C

%post
    apt-get update && apt-get install -y \
        build-essential \
        git

    # Install Python dependencies
    pip install --no-cache-dir \
        hydra-core \
        omegaconf \
        scikit-image \
        tensorboard \
        tqdm

    # Create working directory
    mkdir -p /work

%runscript
    exec bash "$@"

%test
    python -c "import torch; print(f'PyTorch {torch.__version__}')"

%labels
    Author Your Name
    Version v0.1.0
    Project YourProject
```

**Key sections**:

-   `%post`: Install packages, runs as root
-   `%environment`: Set env vars (applies at runtime)
-   `%test`: Validation checks
-   `%runscript`: Default command when container executes

### Step 2: Create Build Batch Script

```bash
#!/usr/bin/env bash
#SBATCH --job-name=build_container
#SBATCH --output=build.out
#SBATCH --error=build.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G

# Setup Singularity cache
export SINGULARITY_TMPDIR=$HOME/.singularity/tmp
export SINGULARITY_CACHEDIR=$HOME/.singularity/cache
mkdir -p $SINGULARITY_CACHEDIR $SINGULARITY_TMPDIR

# Build container
singularity build --fakeroot \
    output.sif \
    definition.def
```

**Important**: `--fakeroot` allows build without root privileges; `--localimage` uses cached images.

### Step 3: Submit Build

```bash
sbatch build_script.sh
```

Monitor with `squeue --me` or check `.out` file.

### Step 4: Test Container

```bash
# Interactive test on compute node
srun --gres=gpu:1 singularity exec --nv output.sif python -c "import torch; print(torch.cuda.is_available())"

# Expected output: True
```

---

## 2\. Running Jobs on AI Cloud

### Job Submission Basics

Jobs submitted via **`srun`** (interactive) or **`sbatch`** (batch script).

#### Default Access

```bash
# Defaults: --account=aau, --qos=normal, --partition=prioritized
srun hostname
```

Verify with: `srun bash -c 'env | grep SLURM'`

### Access Modes & Resource Quotas

#### Normal QOS (Default)

```bash
# Default settings, no flags needed
srun script.py

# Or explicitly:
srun --qos=normal --account=aau python script.py
```

**Limits**: Standard quota, limited concurrent jobs.

#### Unprivileged Access (Multiple Jobs)

```bash
# Allows infinite simultaneous jobs (preemptable)
srun --qos=unprivileged --time=2-00:00:00 python train.py
```

**Key**: Jobs can be interrupted if higher-priority requests arrive; requeued automatically. Good for experimental/long-running work.

#### Deadline Access (14-day boost)

```bash
# Request deadline resources (requires approval via serviceportal.aau.dk)
srun --account=deadline --time=14-00:00:00 python script.py
```

**Apply at**: [AI Cloud: Request deadline resources](https://aau.service-now.com/serviceportal?id=sc_cat_item&sys_id=22a816638322be5053711d447daad379)

-   12 extra concurrent jobs + 12 extra GPUs for 14 days
-   Processed same day
-   Cannot reapply for 14 days after grant expires

### Resource Allocation

```bash
# 8 CPU cores, 32GB RAM, 2 GPUs
srun --cpus-per-task=8 --mem=32G --gres=gpu:2 python script.py

# Using sbatch with batch script
sbatch --cpus-per-task=8 --mem=32G --gres=gpu:2 job.sh
```

**Common allocations**:

-   Single GPU training: `--gres=gpu:1`
-   Multi-GPU: `--gres=gpu:2` or `--gres=gpu:A100:2` (specific GPU type)

### Time Limits

```bash
# 2-hour limit
srun --time=02:00:00 python script.py

# Format: Days-HH:MM:SS
srun --time=1-12:30:00 python long_job.py
```

### Batch Script Template

```bash
#!/usr/bin/env bash
#SBATCH --job-name=train_model
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=prioritized
#SBATCH --account=aau
#SBATCH --qos=normal

# Load container
CONTAINER=/path/to/container.sif

# Run training
singularity exec --nv $CONTAINER \
    python scripts/train.py \
        --config configs/training/default.yaml \
        training.seed=42
```

Submit: `sbatch job_script.sh`

---

## 3\. Monitoring & Inspecting Jobs

### View Job Queue

```bash
# All jobs (entire cluster)
squeue

# Your jobs only
squeue --me

# Detailed output with all fields
squeue -o "%all"
```

**Output columns**:

-   `JOBID`: Unique job identifier
-   `PARTITION`: Cluster partition (e.g., `prioritized`, `batch`)
-   `NAME`: Job name (user-specified)
-   `USER`: Submitting user
-   `ST`: Job state (`R`\=running, `PD`\=pending, `CA`\=cancelled, `CD`\=completed)
-   `TIME`: Elapsed runtime
-   `NODES`: Number of compute nodes
-   `NODELIST(REASON)`: Node(s) or why pending (e.g., `(Dependency)`, `(Resources)`)

### Jobs Example

```
JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
31623     batch     train xxxxxxxx  R    6:45:14      1 i256-a10-10
31694     batch singular yyyyyyyy  R      24:20      1 i256-a40-01
31502 prioritiz runQHGK. zzzzzzzz PD       0:00      1 (Dependency)
```

**Reading**: Job 31623 running on GPU node i256-a10-10, 6h 45m elapsed. Job 31502 pending due to dependency.

### Check Compute Node Status

```bash
# View node health, load, GPU availability
sinfo

# More detail
sinfo -lN
```

### Check GPU Utilization (On Compute Node)

```bash
# From within a running job:
srun nvidia-smi

# Continuous monitoring (10s interval)
watch -n 10 nvidia-smi
```

### Job Information

```bash
# Full details of specific job
scontrol show job JOBID

# Job status
sacct -j JOBID
```

### Cancel Job

```bash
scancel JOBID

# Cancel all your jobs
scancel --user=$USER
```

---

## 4\. Container Execution Examples

### Single GPU Training

```bash
# Via sbatch
sbatch --gres=gpu:1 train_job.sh

# Contents of train_job.sh:
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

singularity exec --nv container.sif python scripts/train.py
```

### Multi-GPU with PyTorch

```bash
# Request 2 GPUs
srun --gres=gpu:2 singularity exec --nv container.sif \
    python -m torch.distributed.launch \
        --nproc_per_node=2 \
        scripts/train.py
```

### Interactive Development

```bash
# Allocate resources and drop into shell
srun --gres=gpu:1 --pty singularity shell --nv container.sif

# Then inside container:
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 5\. Key Configurations & Best Practices

### Environment Setup

```bash
# Set Singularity cache (before builds)
export SINGULARITY_CACHEDIR=$HOME/.singularity/cache
export SINGULARITY_TMPDIR=$HOME/.singularity/tmp
mkdir -p $SINGULARITY_CACHEDIR $SINGULARITY_TMPDIR
```

### Verify Slurm Defaults

```bash
# Check environment variables set by Slurm
srun bash -c 'env | grep SLURM'

# Look for:
# SLURM_JOB_PARTITION=prioritized
# SLURM_JOB_ACCOUNT=aau
# SLURM_JOB_QOS=normal
```

### Container Best Practices

1.  **Use `--fakeroot`** when building (avoids root requirement)
2.  **Pin versions** in definition files (reproducibility)
3.  **Test locally** before submitting large jobs
4.  **Use `--nv`** flag with `singularity exec` for GPU support

### Resource Estimation

-   **GPU memory**: Check model size vs. GPU VRAM (A100: 40GB, A10: 24GB)
-   **CPU/RAM**: Rule of thumb: 1 CPU per GPU + 4-8GB RAM per GPU
-   **Time limits**: Add 20% buffer; check job logs for actual runtime

---

## 6\. Troubleshooting

Issue

Solution

Container build fails

Check `.out` file; ensure sandbox space available

Job stuck in `PD` (pending)

Check `(Reason)` in squeue; may be waiting for resources or dependencies

Out of GPU memory

Reduce batch size; request larger GPU; check `nvidia-smi`

Container not found

Use absolute path: `/path/to/container.sif`; check permissions

Test section fails

Review `%test` in def file; test manually inside container first

---

## References

-   **AI Cloud Home**: [https://hpc.aau.dk/ai-cloud/](https://hpc.aau.dk/ai-cloud/)
-   **Building Containers**: [https://hpc.aau.dk/ai-cloud/additional-guides/building-your-own-container-image/](https://hpc.aau.dk/ai-cloud/additional-guides/building-your-own-container-image/)
-   **Running Jobs**: [https://hpc.aau.dk/ai-cloud/getting-started/run-jobs/](https://hpc.aau.dk/ai-cloud/getting-started/run-jobs/)
-   **Beyond Default Quota**: [https://hpc.aau.dk/ai-cloud/additional-guides/run-jobs-beyond-the-default-qota/](https://hpc.aau.dk/ai-cloud/additional-guides/run-jobs-beyond-the-default-qota/)
-   **Queue Monitoring**: [https://hpc.aau.dk/ai-cloud/additional-guides/checking-the-queue/](https://hpc.aau.dk/ai-cloud/additional-guides/checking-the-queue/)
-   **Singularity Docs**: [https://docs.sylabs.io/guides/3.0/user-guide/](https://docs.sylabs.io/guides/3.0/user-guide/)
-   **Slurm Docs**: [https://slurm.schedmd.com/](https://slurm.schedmd.com/)

---

**Last updated**: March 2026