import nibabel as nib
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from medpy.metric.binary import hd95
from cc3d import connected_components as cc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import csv
from sklearn.metrics import precision_recall_curve, auc
import seg_metrics.seg_metrics as sg
from evaluate_metrics import *

# get data
task_path = '/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Task001_wmh/'
gt_path = task_path + 'labelsTs'
#infer_path = task_path + 'inferTs/nnformer_mwsc_t1'
infer_path = '/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Ensemble_wmh'
prob_path = infer_path + '/probs'

print("Loading data...")
label_data = sorted([f for f in os.listdir(gt_path) if os.path.isfile(os.path.join(gt_path,f)) and f[-7:]==".nii.gz"])
infer_data = sorted([f for f in os.listdir(infer_path) if os.path.isfile(os.path.join(infer_path,f)) and f[-7:]==".nii.gz"])
prob_data = sorted([f for f in os.listdir(prob_path) if os.path.isfile(os.path.join(prob_path,f)) and f[-7:]==".nii.gz" and len(f) == 25]) #25,14

# check if infered labels and test labels match
if label_data == infer_data == prob_data:
    print("Data loaded")
else:
    raise IndexError("Groundtruth data and infered data does not match")

# create file to write results to
fw = open(infer_path + '/evalution.txt', 'w')
fw.write(infer_path.split('/')[-1]+'\n')

# calculate metrics
dice_scores, hd95_score, kappa_scores,rvd_scores = [], [], [], []
tprs, precisions, f1_scores, auc_scores = [], [], [], []

print("Calculating...")
for f in tqdm(infer_data):
    # Load data and convert to numpy
    gt_f_path = os.path.join(gt_path, f)
    gt_img = nib.load(gt_f_path)
    gt = np.asarray(gt_img.dataobj)

    infer_f_path = os.path.join(infer_path, f)
    infer_img = nib.load(infer_f_path)
    infer = np.asarray(infer_img.dataobj)

    prob_f_path = os.path.join(prob_path, f)
    prob_img = nib.load(prob_f_path)
    prob = np.asarray(prob_img.dataobj)
    
    # metrics
    dice_scores.append(cal_dice(gt, infer))
    hd95_score.append(hd95(gt, infer))
    kappa_scores.append(cohen_kappa_score(gt.flatten(),infer.flatten()))
    tpr, precision, f1 = cluster_wise(gt, infer)
    tprs.append(tpr)
    precisions.append(precision)
    f1_scores.append(f1)
    rvd_scores.append(cal_rvd(gt,infer))
    auc_scores.append(sg_metrics(gt_f_path, infer_f_path, infer_f_path[:-25], gt, prob)) #25,14

    # write metrics to the file
    fw.write('*'*20+'\n',)
    fw.write(f[:-7]+'\n') # image name
    fw.write('HD: {:.4f}\n'.format(hd95_score[-1]))
    fw.write('Dice: {:.4f}\n'.format(dice_scores[-1]))
    fw.write('TPR: {:.4f}\n'.format(tprs[-1]))
    fw.write('Precision: {:.4f}\n'.format(precisions[-1]))
    fw.write('F1: {:.4f}\n'.format(f1_scores[-1]))
    fw.write('Kappa: {:.4f}\n'.format(kappa_scores[-1]))
    fw.write('RVD: {:.4f}\n'.format(rvd_scores[-1]))
    fw.write('AUC: {:.4f}\n'.format(auc_scores[-1]))

# calculate average scores
avg_dice = np.mean(dice_scores)
avg_hd = np.mean(hd95_score)
avg_tpr = np.mean(tprs)
avg_precision = np.mean(precisions)
avg_f1 = np.mean(f1_scores)
avg_kappa = np.mean(kappa_scores)
avg_rvd = np.mean(rvd_scores)
avg_auc = np.mean(auc_scores)

print("Dice score:", avg_dice)
print("Hausdorff distance:", avg_hd)
print("Cluster wise:")
print("TPR:", avg_tpr)
print("Precision:", avg_precision)
print("F1 score:", avg_f1)
print("Kappa score:", avg_kappa)
print("RVD score:", avg_rvd)
print("AUC score:", avg_auc)

fw.write('*'*20+'\n',)
fw.write('Average score values:'+'\n')
fw.write('Dice: {:.4f}\n'.format(avg_dice))
fw.write('HD95: {:.4f}\n'.format(avg_hd))
fw.write('TPR: {:.4f}\n'.format(avg_tpr))
fw.write('Precision: {:.4f}\n'.format(avg_precision))
fw.write('F1: {:.4f}\n'.format(avg_f1))
fw.write('Kappa: {:.4f}\n'.format(avg_kappa))
fw.write('RVD: {:.4f}\n'.format(avg_rvd))
fw.write('AUC: {:.4f}\n'.format(avg_auc))
fw.close()

# save metrics for plotting/confidence intervals etc
all_metrics = np.column_stack((dice_scores, hd95_score, kappa_scores, rvd_scores, tprs, precisions, f1_scores, auc_scores))

with open(infer_path + '/metrics.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(['Dice', 'HD95', 'Kappa',' RVD', 'Cluster recall', 'Cluster precision', 'Cluster f1', 'AUC'])
    write.writerows(all_metrics)

# create file to write sg metrics to
df_v = pd.read_csv(f'{infer_path}/sg_metrics.csv')

fw = open(infer_path + '/sg_evaluation.txt', 'w')
fw.write('Average score values:'+'\n')
fw.write('Dice: {:.4f}\n'.format(df_v.loc[:, 'dice'].mean()))
fw.write('HD95: {:.4f}\n'.format(df_v.loc[:, 'hd95'].mean()))
fw.write('Precision: {:.4f}\n'.format(df_v.loc[:, 'precision'].mean()))
fw.write('Recall: {:.4f}\n'.format(df_v.loc[:, 'recall'].mean()))
fw.write('FPR: {:.4f}\n'.format(df_v.loc[:, 'fpr'].mean()))
fw.write('FNR: {:.4f}\n'.format(df_v.loc[:, 'fnr'].mean()))
fw.write('Volume Similarity: {:.4f}\n'.format(df_v.loc[:, 'vs'].mean()))
