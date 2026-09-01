---
name: singularity
description: "Create and manage Singularity containers for HPC. Use when you need reproducible Singularity 3.5 definition-file builds, validation, and execution in batch jobs, including GPU workloads and Slurm integration."
argument-hint: "HPC goal, base image/source, scheduler constraints, and GPU yes/no"
user-invocable: true
---
# Singularity

## What This Skill Produces

-   A repeatable HPC workflow to create and manage Singularity containers.
-   A decision path for definition-file-first builds, with pull/sandbox branches when needed.
-   A validated image and execution pattern for batch and GPU jobs.

## When To Use

-   You need a new `.sif` image from Docker, Library, or a definition file.
-   You need to maintain an HPC-oriented Singularity workflow.
-   You need to run one-shot commands in batch jobs or GPU workloads.
-   You need reproducibility checks for research and cluster execution.

## Required Inputs

-   Objective: what the container must do.
-   Base source: `library://`, `docker://`, local image, or definition file.
-   Build mode: privileged (`sudo`) or unprivileged (`--fakeroot` when available).
-   Runtime needs: bind mounts, env vars, GPU (`--nv` / `--rocm`), and scheduler constraints.

## Decision Flow

1.  If the goal is production or publication, build from a definition file.
2.  If the goal is fast prototyping, pull an existing image, then convert flow to a definition-file build.
3.  If debugging package steps, use a `--sandbox` iteration branch, then rebuild final `.sif` from definition file.
4.  If root access is unavailable on the cluster, use `--fakeroot` if configured, otherwise use remote builder.
5.  If workload requires GPUs, include runtime invocation with `--nv` (NVIDIA) or `--rocm` (AMD).

## Procedure

1.  Preflight
    -   Confirm installation and version: `singularity version`
    -   Confirm docs and behavior target Singularity 3.5 workflow.
    -   Check available capabilities and remotes if needed:
        -   `singularity remote list`
        -   `singularity cache list`
2.  Acquire or define the base
    -   Pull existing image:
        -   `singularity pull app.sif library://<collection>/<container>:<tag>`
        -   `singularity pull app.sif docker://ubuntu:22.04`
    -   Or create a definition file if custom build is required.
3.  Author the definition file
    -   Header must include `Bootstrap` and matching keys such as `From`.
    -   Prefer `%files` over `%setup` for host-to-container copies.
    -   Use `%post` for package install and build-time setup.
    -   Use `%environment` for runtime env vars.
    -   Use `%runscript` for default run behavior.
    -   Use `%test` to enforce validation during build.
    -   Add `%labels` and `%help` for discoverability and documentation.
4.  Build image
    -   Standard: `sudo singularity build app.sif app.def`
    -   Rootless when supported: `singularity build --fakeroot app.sif app.def`
    -   Sandbox for iteration: `sudo singularity build --sandbox app.sandbox app.def`
    -   Skip tests only when justified: `--notest`
5.  Validate image
    -   Metadata and labels: `singularity inspect app.sif`
    -   Embedded tests: `singularity test app.sif`
    -   Runtime env check: `singularity exec app.sif env | grep -E 'KEY1|KEY2'`
6.  Run and manage
    -   One-shot command (preferred for HPC batch): `singularity exec app.sif <cmd>`
    -   Default entrypoint: `singularity run app.sif`
    -   GPU command:
        -   NVIDIA: `singularity exec --nv app.sif <cmd>`
        -   AMD ROCm: `singularity exec --rocm app.sif <cmd>`
    -   Optional service mode when needed:
        -   `singularity instance start app.sif app1`
        -   `singularity instance list`
        -   `singularity instance stop app1`
7.  HPC and Slurm integration
    -   Minimal Slurm batch example:
        
        ```bash
        #!/bin/bash
        #SBATCH --job-name=singularity-job
        #SBATCH --time=00:30:00
        #SBATCH --cpus-per-task=4
        singularity exec app.sif python train.py
        ```
        
    -   Minimal Slurm GPU batch example:
        
        ```bash
        #!/bin/bash
        #SBATCH --job-name=singularity-gpu
        #SBATCH --gres=gpu:1
        #SBATCH --time=00:30:00
        singularity exec --nv app.sif python train.py --device cuda
        ```
        
    -   Submit with `sbatch job.sh` and capture runtime logs for reproducibility.
8.  Share and lifecycle
    -   Sign/verify when integrity matters:
        -   `singularity sign app.sif`
        -   `singularity verify app.sif`
    -   Push to remote library if needed: `singularity push app.sif <library-ref>`
    -   Keep cache clean in constrained environments: `singularity cache clean`

## Definition File Safety And Quality Rules

-   Treat `%setup` as high risk because it executes on host with elevated privileges.
-   Favor immutable, reproducible builds from definition files over ad-hoc sandbox edits.
-   Keep software under system paths, not transient bind-prone paths such as `/tmp` or `/home`.
-   Ensure `%help`, labels, and tests exist for maintainability.

## Completion Checklist

-   Build source is documented (`app.def` preferred for production).
-   Image builds successfully without manual post-build edits.
-   `singularity test` passes (or skip rationale is recorded).
-   `exec` behavior works in batch mode and expected scheduler environment.
-   GPU path is verified if required (`--nv` or `--rocm`).
-   Required runtime binds/env vars are documented for cluster execution.
-   Optional: image is signed and verification succeeds.

## Troubleshooting Branches

-   Build fails with permissions:
    -   Retry with `sudo` or `--fakeroot` if configured.
-   Build fails in `%post`:
    -   Re-run with simpler package steps and verify network/mirror access.
-   Job passes locally but fails in Slurm:
    -   Compare bind mounts, module loads, and env vars between interactive and batch runs.
-   GPU not detected:
    -   Verify scheduler GPU allocation and add `--nv` or `--rocm` at runtime.
-   Runtime command cannot find files:
    -   Inspect bind mounts and path expectations.
-   Env variables missing at runtime:
    -   Ensure they are in `%environment` or appended via `$SINGULARITY_ENVIRONMENT` in `%post`.

## References

-   [Singularity User Guide 3.5 Index](https://docs.sylabs.io/guides/3.5/user-guide/index.html)
-   [Why Use Singularity? (3.5)](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html#why-use-singularity)
-   [Definition Files (3.5)](https://docs.sylabs.io/guides/3.5/user-guide/definition_files.html)