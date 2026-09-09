#!/usr/bin/env bash
cd $HOME/projects/LuminaScale
singularity exec --nv "$HOME/projects/LuminaScale/luminascale.sif" bash -c "cd $HOME/projects/LuminaScale && NW=${NW:?NW required} V3=${V3:-0} ~/.lumina-env/.pixi/envs/default/bin/python -u scripts/diag_baseline.py"
