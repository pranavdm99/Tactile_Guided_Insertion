#!/bin/bash
#SBATCH --job-name=build_tactile
#SBATCH --output=build_%j.out
#SBATCH --error=build_%j.err
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --account=enpm690-class

module load apptainer
export APPTAINER_CACHEDIR=$HOME/scratch/.apptainer_cache
mkdir -p $APPTAINER_CACHEDIR

apptainer build --fakeroot tactile_insertion.sif tactile_insertion.def
