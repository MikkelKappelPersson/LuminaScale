#!/usr/bin/env bash
cd $HOME/projects/LuminaScale
singularity exec --nv "$HOME/projects/LuminaScale/luminascale.sif" bash -c "cd $HOME/projects/LuminaScale && NGPU=${NGPU:?NGPU required} NW=${NW:-4} ~/.lumina-env/.pixi/envs/default/bin/python -u scripts/diag_ddp.py"
