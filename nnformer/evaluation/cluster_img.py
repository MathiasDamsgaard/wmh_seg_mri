from PIL import Image
import numpy as np
import os
from cc3d import connected_components as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import nibabel as nib

cmap = (mpl.colors.ListedColormap(['black', 'green', 'yellow', 'red']))

path = "/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Task002_t1/labelsTs/"
path2 = "/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Task002_t1/inferTs/nnformer_wmh_t1/"



gt_img = nib.load(os.path.join(path, "sub-lisa119_ses-01.nii.gz"))
infer_img = nib.load(os.path.join(path2, "sub-lisa119_ses-01.nii.gz"))
image1 = np.asarray(gt_img.dataobj)
image2 = np.asarray(infer_img.dataobj)

def clusterwise_comp(label, pred):
    # TP
    TP_mask = np.logical_and(label, pred)
    out_tp = np.int64(TP_mask)
    # FP
    FP_mask = np.logical_xor(pred, out_tp)
    out_fp = np.int64(FP_mask)
    # FN
    FN_mask = np.logical_xor(label, out_tp)
    out_fn = np.int64(FN_mask)

    mask = TP_mask+FP_mask*2+FN_mask*3

    mask_nii= nib.Nifti1Image(mask.astype(np.int32), affine=gt_img.affine)

    nib.save(mask_nii, "/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Task002_t1/voxel.nii.gz")

    # connected components
    labels = cc(label, connectivity=26)
    preds = cc(pred, connectivity=26)


    lab_cluster_count = len(np.unique(labels))-1
    pred_cluster_count = len(np.unique(preds))-1

    # True positives for cluster
    tmp_tp = labels * out_tp
    tp_cluster_idx = np.unique(tmp_tp)
    tp_cluster = len(tp_cluster_idx)-1
    tp_cluster_mask = np.isin(labels, tp_cluster_idx[1:])


    # False positives for cluster
    tmp_fp = preds * out_fp
    fp_cluster_idx = np.unique(tmp_fp)

    tmp_fp = preds * out_tp
    fp_cluster_idx = [i for i in fp_cluster_idx if i not in np.unique(tmp_fp)]
    fp_cluster = len(fp_cluster_idx)

    fp_cluster_mask = np.isin(preds, fp_cluster_idx)

    # False negatives for cluster
    tmp_fn = labels * out_fn
    fn_cluster_idx = np.unique(tmp_fn)

    fn_cluster_idx = [i for i in fn_cluster_idx if i not in np.unique(tmp_tp)]
    fn_cluster = len(fn_cluster_idx)

    fn_cluster_mask = np.isin(labels, fn_cluster_idx)

    cluster_mask = tp_cluster_mask+fp_cluster_mask*2+fn_cluster_mask*3


    mask_nii= nib.Nifti1Image(cluster_mask.astype(np.int32), affine=gt_img.affine)

    print(np.unique(np.asarray(mask_nii.dataobj)))

    nib.save(mask_nii, "/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Task002_t1/cluster.nii.gz")

    # TPR (recall) cluster-wise (of all actual clusters, what is the ratio which is actually found)
    tpr = (tp_cluster)/(lab_cluster_count)

    # Precision clusterwise (of all predicted clusters, what is the ratio of actual clusters)
    precision = tp_cluster/(tp_cluster+fp_cluster)
    
    # F1-score
    f1 = 2*tp_cluster/(2*tp_cluster+fp_cluster+fn_cluster)

    # FPR cluster-wise
    # Non-sensical for clusterwise since there are no true negative clusters


    return tpr, precision, f1


print(clusterwise_comp(image1,image2))