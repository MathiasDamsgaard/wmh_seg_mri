#!/bin/bash

while getopts 'c:n:t:m:f:r:p' OPT; do
    case $OPT in
        c) cuda=$OPTARG;;
        n) name=$OPTARG;;
		t) task=$OPTARG;;
        m) model=$OPTARG;;
        f) folds=$OPTARG;;
        r) train="true";;
        p) predict="true";;
        
    esac
done

REPO_PATH="/home/mathiascd/wmh_seg_mri/"
DATA_PATH="/home/mathiascd/DATASET/nnFormer_raw/nnFormer_raw_data/Task001_wmh/"

export nnFormer_raw_data_base="/home/mathiascd/DATASET/nnFormer_raw"
export nnFormer_preprocessed="/home/mathiascd/DATASET/nnFormer_preprocessed"
export RESULTS_FOLDER="/home/mathiascd/DATASET/nnFormer_results"

if ${train}
then
	cd "${REPO_PATH}"
	CUDA_VISIBLE_DEVICES=${cuda} nnFormer_train ${model} nnFormerTrainerV2_${name} ${task} ${folds}
fi

if ${predict}
then
	cd "${DATA_PATH}"
	CUDA_VISIBLE_DEVICES=${cuda} nnFormer_predict -i imagesTs -o inferTs/${name} -m ${model} -t ${task} -f ${folds} -chk model_best -tr nnFormerTrainerV2_${name}

    cd "${REPO_PATH}"
	python ./nnformer/inference_wmh.py ${name}
fi
