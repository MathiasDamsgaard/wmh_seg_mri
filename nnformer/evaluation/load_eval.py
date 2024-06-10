import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
from cc3d import connected_components as cc
import csv



path = '/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Task001_wmh/'


loads = []
cluster_loads = []

labelTr = sorted([os.path.join('labelsTr',f) for f in os.listdir(path+'labelsTr') if os.path.isfile(os.path.join(path+'labelsTr',f)) and f[-6:]=="nii.gz"])
labelTs = sorted([os.path.join('labelsTs',f) for f in os.listdir(path+'labelsTs') if os.path.isfile(os.path.join(path+'labelsTs',f)) and f[-6:]=="nii.gz"])
inter = [os.path.join('labelsTr','sub-lisa105_ses-01.nii.gz'), os.path.join('labelsTr','sub-lisa116_ses-01.nii.gz'),os.path.join('labelsTs','sub-lisa127_ses-01.nii.gz'),os.path.join('labelsTr','sub-lisa137_ses-01.nii.gz'), os.path.join('labelsTr','sub-lisa140_ses-01.nii.gz'), os.path.join('labelsTr','sub-lisa202_ses-01.nii.gz'), os.path.join('labelsTr','sub-lisa326_ses-01.nii.gz')]

for dataset in [inter]:
    for f in tqdm(dataset):
        img = nib.load(os.path.join(path,f))
        img = np.asarray(img.dataobj)

        loads.append(img.sum())

        clusters = cc(img, connectivity=26)
        cluster_loads.append(len(np.unique(clusters))-1)



all_metrics = np.column_stack((loads, cluster_loads))

with open('/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data//load_data.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(['Load', 'Cluster load'])
    write.writerows(all_metrics)


print('Average WMH load:', np.mean(loads))
print('Average amount of clusters', np.mean(cluster_loads))