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

REPO_PATH="/home/mathiascd/nnFormer/"
DATA_PATH="/home/mathiascd/DATASET/nnFormer_raw/nnFormer_raw_data/Task003_tumor/"

if ${train}
then
	
	cd "${REPO_PATH}"
	CUDA_VISIBLE_DEVICES=${cuda} nnFormer_train ${model} nnFormerTrainerV2_${name} ${task} ${folds}
fi

if ${predict}
then

	cd "${DATA_PATH}"
	CUDA_VISIBLE_DEVICES=${cuda} nnFormer_predict -i imagesTs -o inferTs/${name} -m ${model} -t ${task} -f 0 -chk model_best -tr nnFormerTrainerV2_${name}

    cd "${REPO_PATH}"
	python inference_tumor.py ${name}
fi
