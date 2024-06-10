import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import os
import nibabel as nib


def boxplot(data, metric):
    sns.set_theme(style="darkgrid")
    sns.boxplot(data)
    plt.axvline(x=3.5, color='grey', linestyle='--')

    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.show()


def scatter_plot(data, metric):
    sns.set_theme(style="darkgrid")
    ax = sns.scatterplot(data, x=metric, y='Load')
    sns.regplot(data, x=metric, y="Load", scatter=False, ax=ax)

    plt.xlabel(metric)
    plt.ylabel('WMH load (log)')
    plt.show()

def pr_curve_plot(ground_truth, predictions):
    model_names = ["Basic nnFormer", "Deep nnFormer", 'Spatial nnFormer', 'Ensemble nnFormer']
    colors = sns.color_palette(n_colors=len(model_names))
    sns.set_theme(style="darkgrid")


    y_true = np.concatenate([g.flatten() for g in ground_truth])

    for i, pred in enumerate(predictions):
        y_scores = np.concatenate([p.flatten() for p in pred])
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        auc_pr = auc(recall, precision)
        print(auc_pr)
    
        plt.plot(recall, precision, label=f'{model_names[i]} (AUC-PR={auc_pr:.3f})', color=colors[i])
        

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()
    


bianca = pd.read_csv('/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Extern_models/BIANCA/metrics.csv')
nnunet = pd.read_csv('/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Extern_models/nnU-Net/metrics.csv')
unetr = pd.read_csv('/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Extern_models/UNETR/metrics.csv')
vosht = pd.read_csv('/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Extern_models/VoSHT/metrics.csv')

nds = pd.read_csv('/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Task001_wmh/inferTs/nnformer_wmh_nds/metrics.csv')
ds = pd.read_csv('/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Task001_wmh/inferTs/nnformer_wmh_ds/metrics.csv')
t1 = pd.read_csv('/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Task002_t1/inferTs/nnformer_wmh_t1/metrics.csv')
ensemb = pd.read_csv('/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Ensemble_wmh/metrics.csv')


dice = {'BIANCA':bianca['Dice'], 'nnU-Net': nnunet['Dice'], 'UNETR':unetr['Dice'], 'VoSHT':vosht['Dice'], 'Basic nnFormer': nds['Dice'], 'Deep nnFormer': ds['Dice'], 'Spatial nnFormer': t1['Dice'], 'Ensemble nnFormer': ensemb['Dice']}
hd = {'BIANCA':bianca['HD95'], 'nnU-Net': nnunet['HD95'], 'UNETR':unetr['HD95'], 'VoSHT':vosht['HD95'], 'Basic nnFormer': nds['HD95'], 'Deep nnFormer': ds['HD95'], 'Spatial nnFormer': t1['HD95'], 'Ensemble nnFormer': ensemb['HD95']}

boxplot(dice, 'Dice Score')
boxplot(hd, 'HD95')

lab_p = '/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Task003_MWSC/labelsTs/'
nds_p = '/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Task003_MWSC/inferTs/nnformer_mwsc_nds/probs/'
ds_p = '/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Task003_MWSC/inferTs/nnformer_mwsc_ds/probs/'
t1_p = '/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Task004_MWSC-t1/inferTs/nnformer_mwsc_t1/probs/'
ens_p = '/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Ensemble_mwsc/probs/'


label = [np.asarray(nib.load(os.path.join(lab_p,f)).dataobj) for f in sorted(os.listdir(lab_p)) if f[-7:]==".nii.gz"]
prob_files = [[np.asarray(nib.load(os.path.join(nds_p,f)).dataobj) for f in sorted(os.listdir(nds_p)) if f[-7:]==".nii.gz" and len(f)==14],
              [np.asarray(nib.load(os.path.join(ds_p,f)).dataobj) for f in sorted(os.listdir(ds_p)) if f[-7:]==".nii.gz" and len(f)==14],
              [np.asarray(nib.load(os.path.join(ens_p,f)).dataobj) for f in sorted(os.listdir(t1_p)) if f[-7:]==".nii.gz" and len(f)==14],
              [np.asarray(nib.load(os.path.join(t1_p,f)).dataobj) for f in sorted(os.listdir(ens_p)) if f[-7:]==".nii.gz" and len(f)==14]]

pr_curve_plot(label, prob_files)


load = pd.read_csv('/mnt/projects/whmseg/nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/load_data.csv')

l_dice = {'Dice Score': ensemb['Dice'], 'Load': np.log(load['Load'])}
l_hd = {'HD95': ensemb['HD95'], 'Load': np.log(load['Load'])}

scatter_plot(l_dice, 'Dice Score')
scatter_plot(l_hd, 'HD95')


