#!/bin/bash

#SBATCH --partition=Copy
#SBATCH --output='/mnt/scratch/projects/wmh_seg/slurm_%A_%a.out'
#SBATCH --chdir='/mnt/scratch/projects/wmh_seg/'
#SBATCH --mail-type=ALL        
#SBATCH --mail-user=s214647@dtu.dk
#SBATCH --dependency afterok:!!!CHANGE TO PREVIOUS JOBID!!!

DIR=/mnt/scratch/projects/wmh_seg/
rsync -vah ${DIR}+data ${DIR}