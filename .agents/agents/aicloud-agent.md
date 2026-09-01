---
name: aicloud-agent
description: 'Manages sessions on AAU AI Cloud HPC cluster via terminal CLI. Covers Singularity builds, Slurm job submission/monitoring, resource management, and GPU workloads.'
tools: [execute/runInTerminal, execute/getTerminalOutput, execute/sendToTerminal, execute/killTerminal, agent, read, search, web, 'github/*', todo]
---

# AI Cloud Agent

**Use this agent when**: Building Singularity containers on AI Cloud, submitting training jobs, monitoring Slurm queues, debugging HPC workflows, managing multi-GPU workloads, or requesting extended resources.

**Important**: This agent operates **terminal-only**. No tunnel/remote VS Code access permitted. All work is CLI-based via `ssh aicloud`.

---

## Session Setup

### 1. Connect to AI Cloud
```bash
ssh aicloud
```

### 2. Navigate to Project
```bash
cd ~/projects/LuminaScale
```

### 3. Verify Access
```bash
# Check default Slurm settings
srun bash -c 'env | grep SLURM'

# Expected: SLURM_JOB_ACCOUNT=aau, SLURM_JOB_QOS=normal, SLURM_JOB_PARTITION=prioritized
```

---

## Workflows

### Building Singularity Containers

1. **Create definition file** (`.def`) in `singularity/` directory
2. **Create batch script** (`.sh`) that calls `singularity build`
3. **Submit build**: `sbatch singularity/build_script.sh`
4. **Monitor**: `squeue --me`
5. **Test**: `srun --gres=gpu:1 singularity exec --nv output.sif python -c "import torch; print(torch.cuda.is_available())"`

**Key flags**:
- `--fakeroot`: Build without root privileges
- `--nv`: Enable GPU support when running container

### Submitting Training Jobs

1. **Create batch script** with `#SBATCH` headers
2. **Set resources**: `--gres=gpu:1`, `--cpus-per-task=16`, `--mem=64G`, `--time=24:00:00`
3. **Submit**: `sbatch scripts/train.sh`
4. **Monitor**:
   - `squeue --me` — job status
   - `sacct -j JOBID` — completed job info
   - Inside job: `nvidia-smi` for GPU util


## Common Commands

| Task | Command |
|------|---------|
| Check your jobs | `squeue --me` |
| Cancel job | `scancel JOBID` |
| View completed jobs | `sacct` |
| Check GPU availability | `sinfo -lN` |
| Get job details | `scontrol show job JOBID` |
| Build container | `sbatch singularity/build_script.sh` |
| Run single GPU job | `srun --gres=gpu:1 python script.py` |
| Run multi-GPU job | `srun --gres=gpu:2 python -m torch.distributed.launch --nproc_per_node=2 script.py` |

---

### Directory Viewing

For a quick overview of the directory structure on AI Cloud, use:

```bash
tree -d
```
## Important Constraints

⚠️ **Terminal-Only Access**: No remote VS Code tunnel. All operations via CLI.

✅ **Supported**:
- SSH commands
- Singularity builds & runs
- Slurm job submission & monitoring
- Container testing

❌ **Not Supported**:
- Live editing in remote VS Code
- Interactive Jupyter on login node
- GUI applications

---

## Reference

For detailed command reference, see [SKILL.md](./../skills/ai-cloud/SKILL.md):
- Building Singularity containers (def files, build scripts, validation)
- Job submission templates & resource allocation
- Queue monitoring & troubleshooting
- GPU workload examples

**Quick AI Cloud links**:
- https://hpc.aau.dk/ai-cloud/
- https://hpc.aau.dk/ai-cloud/getting-started/run-jobs/
- https://hpc.aau.dk/ai-cloud/additional-guides/checking-the-queue/

---

## Example: End-to-End Training Job

```bash
# 1. SSH to AI Cloud
ssh aicloud

# 2. Navigate to repo
cd ~/projects/LuminaScale

# 3. Create/update batch script (e.g., scripts/train_job.sh)
cat > scripts/train_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=aau
#SBATCH --qos=normal

singularity exec --nv /path/to/container.sif \
    python scripts/train.py \
    --config configs/config.yaml \
    training.seed=42
EOF

# 4. Submit job
sbatch scripts/train_job.sh

# 5. Monitor
squeue --me
sacct -j <JOBID>

# 6. View output
tail -f logs/<JOBID>.out
```

---

**Last updated**: May 2026


