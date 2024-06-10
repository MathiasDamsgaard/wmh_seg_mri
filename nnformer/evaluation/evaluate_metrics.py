import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import seg_metrics.seg_metrics as sg
from cc3d import connected_components as cc

def cal_dice(data1, data2):
    # Calculate Dice score
    flat_data1 = data1.flatten()
    flat_data2 = data2.flatten()

    intersection = np.sum(flat_data1 * flat_data2)
    total_volumes = np.sum(flat_data1) + np.sum(flat_data2)
    
    dice = (2.0 * intersection) / total_volumes
    
    return dice

def cal_rvd(label, pred):
    label_volume = label.sum()
    pred_volume = pred.sum()
    
    rvd = (pred_volume-label_volume)/label_volume
    return rvd

def cluster_wise(label, pred):
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

    # plt.imsave('C:\\Users\\lasse\\Desktop\\cluster_test\\voxelwise_plot2.png', np.array(mask).astype(np.uint8), cmap=cmap)

    # connected components
    labels = cc(label, connectivity=26)
    preds = cc(pred, connectivity=26)

    lab_cluster_count = len(np.unique(labels))-1
    pred_cluster_count = len(np.unique(preds))-1

    # True positives for cluster
    tmp_tp = labels * out_tp
    tp_cluster_idx = np.unique(tmp_tp)
    tp_cluster = len(tp_cluster_idx)-1
    #tp_cluster_mask = np.isin(labels, tp_cluster_idx[1:])

    # False positives for cluster
    tmp_fp = preds * out_fp
    fp_cluster_idx = np.unique(tmp_fp)

    tmp_fp = preds * out_tp
    fp_cluster_idx = [i for i in fp_cluster_idx if i not in np.unique(tmp_fp)]
    fp_cluster = len(fp_cluster_idx)

    #fp_cluster_mask = np.isin(preds, fp_cluster_idx)

    # False negatives for cluster
    tmp_fn = labels * out_fn
    fn_cluster_idx = np.unique(tmp_fn)

    fn_cluster_idx = [i for i in fn_cluster_idx if i not in np.unique(tmp_tp)]
    fn_cluster = len(fn_cluster_idx)

    #fn_cluster_mask = np.isin(labels, fn_cluster_idx)
    #cluster_mask = tp_cluster_mask+fp_cluster_mask*2+fn_cluster_mask*3

    # plt.imsave('C:\\Users\\lasse\\Desktop\\cluster_test\\clusterwise_plot2.png', np.array(cluster_mask).astype(np.uint8), cmap=cmap)

    # TPR (recall) cluster-wise (of all actual clusters, what is the ratio which is actually found)
    tpr = (tp_cluster)/(lab_cluster_count)

    # Precision clusterwise (of all predicted clusters, what is the ratio of actual clusters)
    precision = tp_cluster/(tp_cluster+fp_cluster)
    
    # F1-score
    f1 = 2*tp_cluster/(2*tp_cluster+fp_cluster+fn_cluster)

    # FPR cluster-wise
    # Non-sensical for clusterwise since there are no true negative clusters
    return tpr, precision, f1

def sg_metrics(gt_dir, infer_dir, out_dir, gt, probs):
    metrics = sg.write_metrics(labels=[0,1],  # exclude background if needed
                                gdth_path=gt_dir,
                                pred_path=infer_dir,
                                csv_file=f'{out_dir}sg_metrics.csv',
                                metrics=['dice', 'hd95', 'vs', 'precision', 'recall', 'fpr', 'fnr'])
    l, p = gt.flatten(), probs.flatten()
    precision, recall, thresholds2 = precision_recall_curve(l, p)
    auc_score = auc(recall, precision)
    return auc_score