#!/bin/bash
# -*- coding: utf-8 -*-

#SBATCH --output='/mnt/scratch/projects/whmseg/slurm_%A_%a.out'
#SBATCH --chdir='/mnt/scratch/projects/whmseg/'
#SBATCH --mail-type=ALL        
#SBATCH --mail-user=s214647@dtu.dk

#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --mem 32G
#SBATCH --time=7-0

export PATH="/mnt/projects/whmseg/nobackup/miniconda3/envs/nnFormer/bin:$PATH"
export nnFormer_raw_data_base="/mnt/scratch/projects/whmseg/DATASET/nnFormer_raw"
export nnFormer_preprocessed="/mnt/scratch/projects/whmseg/DATASET/nnFormer_preprocessed"
export RESULTS_FOLDER="/mnt/scratch/projects/whmseg/nnFormer_results"

cd /mnt/scratch/projects/whmseg/
CUDA_VISIBLE_DEVICES=0 nnFormer_train 3d_fullres nnFormerTrainerV2_nnformer_t1 2 0

cd /mnt/scratch/projects/whmseg/DATASET/nnFormer_raw/nnFormer_raw_data/Task002_t1/
CUDA_VISIBLE_DEVICES=0 nnFormer_predict -i imagesTs -o inferTs/nnformer_wmh_t1 -m 3d_fullres -t 2 -f 0 -tr nnFormerTrainerV2_nnformer_t1 -chk model_best -z