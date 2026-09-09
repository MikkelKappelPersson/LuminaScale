#!/usr/bin/env bash
cd $HOME/projects/LuminaScale
singularity exec --nv "$HOME/projects/LuminaScale/luminascale.sif" bash -c "cd $HOME/projects/LuminaScale && ~/.lumina-env/.pixi/envs/default/bin/python -u scripts/diag_gpu_train.py"
