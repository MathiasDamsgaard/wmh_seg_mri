import glob
import os
import SimpleITK as sitk
import numpy as np
import argparse
from medpy.metric import binary
from tqdm import tqdm

def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def new_dice(pred,label):
    tp_hard = np.sum((pred == 1).astype(np.float) * (label == 1).astype(np.float))
    fp_hard = np.sum((pred == 1).astype(np.float) * (label != 1).astype(np.float))
    fn_hard = np.sum((pred != 1).astype(np.float) * (label == 1).astype(np.float))
    return 2*tp_hard/(2*tp_hard+fp_hard+fn_hard)
    
def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())
        
def hd(pred,gt):
    if pred.sum() > 0 and gt.sum()>0:
        hd95 = binary.hd95(pred, gt)
        return  hd95
    else:
        return 0
        
def process_label(label):
    return label == 1

def test(fold):
    path='./../../nobackup/DATASET/nnFormer_raw/nnFormer_raw_data/Task001_wmh/'
    label_list=sorted(glob.glob(os.path.join(path,'labelsTs','*nii.gz')))
    infer_list=sorted(glob.glob(os.path.join(path,'inferTs',fold,'*nii.gz')))
    print("loading success...")
    Dice, HD = [], []
    file=path + 'inferTs/'+fold
    fw = open(file+'/dice_pre.txt', 'w')

    for label_path,infer_path in tqdm(zip(label_list,infer_list)):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label,infer = read_nii(label_path),read_nii(infer_path)
        label_p = process_label(label)
        infer_p = process_label(infer)
        Dice.append(dice(infer_p,label_p))
        
        HD.append(hd(infer_p,label_p))
        
        fw.write('*'*20+'\n',)
        fw.write(infer_path.split('/')[-1]+'\n')
        fw.write('hd: {:.4f}\n'.format(HD[-1]))
        fw.write('Dice: {:.4f}\n'.format(Dice[-1]))
        #print('dice: {:.4f}'.format(np.mean(Dice)))

    avg_dsc = np.mean(Dice)
    avg_hd = np.mean(HD)
    fw.write('*'*20+'\n',)
    fw.write('Dice: '+str(avg_dsc)+' '+'\n')
    fw.write('HD: '+str(avg_hd)+' '+'\n')
    fw.close()
    
    #print('Dice'+str(avg_dsc)+' '+'\n')
    #print('HD'+str(avg_hd)+' '+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("fold", help="fold name")
    args = parser.parse_args()
    fold=args.fold
    test(fold)
