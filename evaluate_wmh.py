import nibabel as nib
import numpy as np
import os
# from skimage.metrics import hausdorff_distance

def dice_score(datak1, data2):
    # Calculate Dice score
    flat_data1 = data1.flatten()
    flat_data2 = data2.flatten()

    intersection = np.sum(flat_data1 * flat_data2)
    total_volumes = np.sum(flat_data1) + np.sum(flat_data2)
    
    dice = (2.0 * intersection) / total_volumes
    
    return dice

# get data
path_gt = '/home/rosshaug/DATASET/nnFormer_raw/nnFormer_raw_data/Task001_wmh/labelsTs'
path_infer = '/home/rosshaug/DATASET/nnFormer_raw/nnFormer_raw_data/Task001_wmh/inferTs/nnformer_wmh'

gt_data = sorted([f for f in os.listdir(path_gt) if os.path.isfile(os.path.join(path_gt,f)) and f[-6:]=="nii.gz"])
infer_data = sorted([f for f in os.listdir(path_infer) if os.path.isfile(os.path.join(path_gt,f)) and f[-6:]=="nii.gz"])
print(infer_data)

# check if infered labels and test labels match
if gt_data == infer_data:
    pass
else:
    print("Groundtruth data and infered data does not match")


# calculate metrics
dice_scores = []
hausdorff_distances = []

for f in infer_data:
    # Load data and convert to numpy
    img1 = nib.load(os.path.join(path_gt,f))
    img2 = nib.load(os.path.join(path_infer,f))
    
    data1 = np.asarray(img1.dataobj)
    data2 = np.asarray(img2.dataobj)
    
    # metrics
    dice_scores.append(dice_score(data1,data2))
    #hausdorff_distances()



print(dice_scores)
print(np.mean(dice_scores))
