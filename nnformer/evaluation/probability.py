import numpy as np
import os
from tqdm import tqdm
import nibabel as nib

# Define path variables
# TASK_PATH = '/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Task004_MWSC-t1/'
# INFER_PATH = TASK_PATH + 'inferTs/nnformer_mwsc_t1'
INFER_PATH = '/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Ensemble_wmh/'
PROB_PATH = INFER_PATH + '/probs/'

# Get the files with softmax probabilities
print("Loading data...")
prob_files = sorted([f for f in os.listdir(PROB_PATH) if os.path.isfile(os.path.join(PROB_PATH,f)) and f[-7:]==".nii.gz" and len(f)==25])
print("Data loaded")

print("Calculating...")
probabilities = []
for file in tqdm(prob_files):
    # Load image
    img = nib.load(os.path.join(PROB_PATH, file))

    # Extract values
    wmh_prob = np.asarray(img.dataobj)

    # Calculate background probabilities as the inverse of the wmh probs
    background_prob = np.ones_like(wmh_prob) - wmh_prob

    # Map 0.5 prob to 1 and 1/0 to 0, making a mask showing a gradient from uncertain to certain
    probability = np.ones_like(wmh_prob) - abs(background_prob - wmh_prob)

    # Save the mask as nifti images and set affine to the same as the original
    probability_img = nib.Nifti1Image(probability.astype(np.float32), affine=img.affine)
    nib.save(probability_img, PROB_PATH + file[:-7] + '_mask.nii.gz')