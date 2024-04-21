import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
from skimage.metrics import hausdorff_distance as cal_hd
from medpy.metric.binary import hd95

def cal_dice(data1, data2):
    # Calculate Dice score
    flat_data1 = data1.flatten()
    flat_data2 = data2.flatten()

    intersection = np.sum(flat_data1 * flat_data2)
    total_volumes = np.sum(flat_data1) + np.sum(flat_data2)
    
    dice = (2.0 * intersection) / total_volumes
    
    return dice

# get data
task_path = '/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Task001_wmh/'
gt_path = task_path + 'labelsTs'
infer_path = task_path + 'inferTs/nnformer_wmh_nds'

label_data = sorted([f for f in os.listdir(gt_path) if os.path.isfile(os.path.join(gt_path,f)) and f[-6:]=="nii.gz"])
infer_data = sorted([f for f in os.listdir(infer_path) if os.path.isfile(os.path.join(infer_path,f)) and f[-6:]=="nii.gz"])

# check if infered labels and test labels match
if label_data == infer_data:
    print("Data loaded")
else:
    print("Groundtruth data and infered data does not match")

# create file to write results to
fw = open(infer_path + '/evalution.txt', 'w')

# calculate metrics
dice_scores = []
hausdorff_distances = []
hd95_score = []

print("Calculating...")
for f in tqdm(infer_data):
    # Load data and convert to numpy
    gt_img = nib.load(os.path.join(gt_path,f))
    infer_img = nib.load(os.path.join(infer_path,f))
    
    gt = np.asarray(gt_img.dataobj)
    infer = np.asarray(infer_img.dataobj)
    
    # metrics
    dice_scores.append(cal_dice(gt, infer))
    hausdorff_distances.append(cal_hd(gt, infer))
    hd95_score.append(hd95(gt, infer))

    # write metrics to the file
    fw.write('*'*20+'\n',)
    fw.write(infer_path.split('/')[-1]+'\n')
    fw.write('HD: {:.4f}\n'.format(hausdorff_distances[-1]))
    fw.write('Dice: {:.4f}\n'.format(dice_scores[-1]))

# calculate average scores
avg_dice = np.mean(dice_scores)
avg_hd = np.mean(hausdorff_distances)

print("Dice scores:")
#print(dice_scores)
print(avg_dice)
print("Hausdorff distances:")
#print(hausdorff_distances)
print(avg_hd)

print("HD95: ", np.mean(hd95_score))

fw.write('*'*20+'\n',)
fw.write('Dice: '+str(avg_dice)+' '+'\n')
fw.write('HD: '+str(avg_hd)+' '+'\n')
fw.close()