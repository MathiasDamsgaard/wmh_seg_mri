import nibabel as nib
import numpy as np
import os
from tqdm import tqdm

mwsc = True

if not mwsc:
    #loop over sessions
    for i in range(2):
        data_path = '/mnt/projects/whmseg/nobackup/T1_train_data/ses_0' + str(i+1) + '/Output/'

        # loop over patients
        for pa in tqdm(set([f[8:11] for f in os.listdir(data_path)])):

            # load images and make them into an array
            flair = nib.load(os.path.join(data_path,'sub-lisa' + pa +'_ses-0' + str(i+1) + '_FLAIR_brain_restore.nii.gz'))
            flair_arr = np.asarray(flair.dataobj)

            t1 = nib.load(os.path.join(data_path,'sub-lisa' + pa +'_ses-0' + str(i+1) + '_T1_brain_2FLAIR.nii.gz'))
            t1_arr = np.asarray(t1.dataobj)

            # stack them togehter and save the following image
            flairT1_arr = np.stack((flair_arr,t1_arr), axis = -1)
            
            flairT1 = nib.Nifti1Image(flairT1_arr, affine=flair.affine)
            nib.save(flairT1, '/mnt/projects/whmseg/nobackup/VoSHT/data/FlairT1/sub-lisa' + pa +'_ses-0' + str(i+1) + '_FLAIR_T1.nii.gz')

else:
    DATA_PATH = '/mnt/projects/whmseg/nobackup/MWSC_challenge/mwsc_testset'
    SAVE_PATH = '/mnt/projects/whmseg/nobackup/MWSC_FlairT1'

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # loop over patients
    for pa in tqdm(set([f[4:7] for f in os.listdir(DATA_PATH) if os.path.isfile(DATA_PATH+'/'+f)])):

        # load images and make them into an array
        flair = nib.load(os.path.join(DATA_PATH,'sub-' + pa + '_FLAIR_brain_restore.nii.gz'))
        flair_arr = np.asarray(flair.dataobj)

        t1 = nib.load(os.path.join(DATA_PATH,'sub-' + pa + '_T1_brain_2FLAIR.nii.gz'))
        t1_arr = np.asarray(t1.dataobj)

        # stack them togehter and save the following image
        flairT1_arr = np.stack((flair_arr,t1_arr), axis = -1)

        flairT1 = nib.Nifti1Image(flairT1_arr, affine=flair.affine)
        nib.save(flairT1, os.path.join(SAVE_PATH, 'sub-' + pa + '_FLAIR_T1.nii.gz'))
