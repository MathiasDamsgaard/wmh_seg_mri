##!/mnt/projects/miniconda3/envs/nnFormer/bin/python
# -*- coding: utf-8 -*-

#SBATCH --output='/mnt/scratch/projects/wmh_seg/slurm_%A_%a.out'
#SBATCH --chdir='/mnt/scratch/projects/wmh_seg/'
#SBATCH --mail-type=ALL        
#SBATCH --mail-user=s214647@dtu.dk

#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1

#SBATCH --export nnFormer_raw_data_base="/mnt/scratch/projects/wmh_seg/DATASET/nnFormer_raw"
#SBATCH --export nnFormer_preprocessed="/mnt/scratch/projects/wmh_seg/DATASET/nnFormer_preprocessed"
#SBATCH --export RESULTS_FOLDER="/mnt/scratch/projects/wmh_seg/DATASET/nnFormer_results"

<actual code generating data in /mnt/scratch/projects/wmh_seg/data>