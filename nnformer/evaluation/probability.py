import numpy as np
from sklearn.metrics import auc, precision_recall_curve
import os
from tqdm import tqdm
import nibabel as nib

# Define path variables
PATH = '/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Task001_wmh/inferTs/nnformer_wmh'

# Get the files with softmax probabilities
print("Loading data...")
prob_files = sorted([f for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH,f)) and f[-4:]==".npz"])
print("Data loaded")

print("Calculating...")
for file in tqdm(prob_files):
    # Load data
    infer = np.load(os.path.join(PATH, file))['softmax']
    