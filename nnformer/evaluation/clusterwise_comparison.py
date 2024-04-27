import numpy as np
from cc3d import connected_components as cc
import glob
import nibabel as nib
#import seaborn as sns
from sklearn.metrics import f1_score

def cluster_wise(label_d, pred_d):
    # for file_i in glob.glob(pth):
    #     sesj = file_i.split('/')[-1].split('_')[1]
    #     subj = file_i.split('/')[-1].split('_')[-1].split('.')[0]
    #     label = nib.load(file_i)
    #     str_p = file_i.replace('label', 'pred')
    #     pred = nib.load(str_p)
    #     affine_d = label.affine
    #     label_d = np.array(label.dataobj)
    #     pred_d = np.array(pred.dataobj)
    
    # TP
    TP_mask = np.logical_and(pred_d, label_d)
    out_tp = np.int64(TP_mask)
    
    # FP
    FP_mask = np.logical_xor(pred_d, out_tp)
    out_fp = np.int64(FP_mask)

    # Labeling
    labels = cc(label_d)
    preds = cc(pred_d)

    # Count label groups
    lab_v, lab_c = np.unique(labels, return_counts=True)
    _, pred_c = np.unique(preds, return_counts=True)

    # TPR cluster-wise
    tmp_t = labels * out_tp
    _, tmp_tp = np.unique(tmp_t, return_counts=True)
    tpr = (len(tmp_tp)-1)/(len(lab_c)-1)
    
    # FPR cluster-wise
    fpr_c = cc(out_fp)
    _, tmp_fp = np.unique(fpr_c, return_counts=True)
    fpr = (len(tmp_fp)-1)/(len(pred_c)-1)
    # print("number cc", len(tmp_fp))
    # print("number in pred", len(pred_c))
    # print("number on label", len(lab_c))
    # print("true", len(tmp_tp))
    # return
    # Precision cluster-wise
    # fp_cl = 0
    # for i in range(1, len(lab_v)):
    #     loc_l = np.where(labels == lab_v[i])
    #     if np.all(preds[loc_l]==0):
    #         fp_cl += 1
    
    precision = (len(tmp_tp)-1)/(len(pred_c)-1)
    #precision = (len(tmp_tp)-1)/(len(tmp_tp)-1+fp_cl)
    
    # F1-score
    f1 = 2*tpr*precision/(tpr+precision)

    return tpr, fpr, f1


# AVD (average volume difference)
import SimpleITK as sitk

def getAVD(testImage, resultImage):   
    """Volume statistics."""
    # Compute statistics of both images
    testStatistics   = sitk.StatisticsImageFilter()
    resultStatistics = sitk.StatisticsImageFilter()
    
    testStatistics.Execute(testImage)
    resultStatistics.Execute(resultImage)
        
    return float(abs(testStatistics.GetSum() - resultStatistics.GetSum())) / float(testStatistics.GetSum()) * 100
