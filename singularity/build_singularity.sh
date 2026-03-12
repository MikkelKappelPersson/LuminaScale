#!/usr/bin/env bash

#SBATCH --job-name=build_luminascale
#SBATCH --output=build_luminascale.out
#SBATCH --error=build_luminascale.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G

export SINGULARITY_TMPDIR=$HOME/.singularity/tmp
export SINGULARITY_CACHEDIR=$HOME/.singularity/cache
mkdir -p $SINGULARITY_CACHEDIR $SINGULARITY_TMPDIR

# The path to the definition file
input_def="singularity/luminascale.def"

# The resulting container image
output_sif="luminascale.sif"

singularity build --fakeroot $output_sif $input_def
