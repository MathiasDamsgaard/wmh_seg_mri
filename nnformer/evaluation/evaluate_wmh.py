import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
from skimage.metrics import hausdorff_distance as cal_hd
from sklearn.metrics import cohen_kappa_score as cal_kappa
from medpy.metric.binary import hd95
#from scipy.ndimage import label
from clusterwise_comparison import cluster_wise, getAVD
from sklearn.metrics import confusion_matrix

def cal_dice(data1, data2):
    # Calculate Dice score
    flat_data1 = data1.flatten()
    flat_data2 = data2.flatten()

    intersection = np.sum(flat_data1 * flat_data2)
    total_volumes = np.sum(flat_data1) + np.sum(flat_data2)
    
    dice = (2.0 * intersection) / total_volumes
    
    return dice

# get data
task_path = '/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Task003_MICCAI/'
gt_path = task_path + 'labelsTs'
infer_path = task_path + 'inferTs/nnformer_wmh'

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
dice_scores, hausdorff_distances, hd95_score = [], [], []
kappa_scores, tprs, fprs, f1_scores, avd_scores = [], [], [], [], []

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
    #kappa_scores.append(cal_kappa(gt, infer))
    tpr, fpr, f1 = cluster_wise(gt, infer)
    tprs.append(tpr)
    fprs.append(fpr)
    f1_scores.append(f1)
    #avd_scores.append(getAVD(gt, infer))

    # write metrics to the file
    fw.write('*'*20+'\n',)
    fw.write(infer_path.split('/')[-1]+'\n')
    fw.write('HD: {:.4f}\n'.format(hausdorff_distances[-1]))
    fw.write('Dice: {:.4f}\n'.format(dice_scores[-1]))
    #fw.write('CK: {:.4f}\n'.format(kappa_scores[-1]))
    fw.write('TPR: {:.4f}\n'.format(tprs[-1]))
    fw.write('FPR: {:.4f}\n'.format(fprs[-1]))
    fw.write('F1: {:.4f}\n'.format(f1_scores[-1]))
    #fw.write('AVD: {:.4f}\n'.format(avd_scores[-1]))

# calculate average scores
avg_dice = np.mean(dice_scores)
avg_hd = np.mean(hausdorff_distances)
#avg_ck = np.mean(kappa_scores)
avg_tpr = np.mean(tprs)
avg_fpr = np.mean(fprs)
avg_f1 = np.mean(f1)
#avg_avd = np.mean(avd_scores)

print("Dice scores:", avg_dice)
print("Hausdorff distances:", avg_hd)
#print("Cohen kappa:", avg_ck)
print("Cluster wise:")
print("TPR:", avg_tpr)
print("FPR:", avg_fpr)
print("F1 score:", avg_f1)
#print("Average volume difference:", avg_avd)
print("HD95: ", np.mean(hd95_score))

fw.write('*'*20+'\n',)
fw.write('Dice: '+str(avg_dice)+' '+'\n')
fw.write('HD: '+str(avg_hd)+' '+'\n')
fw.write('TPR: '+str(avg_tpr)+' '+'\n')
fw.write('FPR: '+str(avg_fpr)+' '+'\n')
fw.write('F1: '+str(avg_f1)+' '+'\n')
#fw.write('AVD: '+str(avg_avd)+' '+'\n')
fw.close()