# Imports
from batchgenerators.utilities.file_and_folder_operations import *
from utils import generate_dataset_json
from nnformer.paths import nnFormer_raw_data
from tqdm import tqdm
import os
import shutil

if __name__ == '__main__': 
    base = os.getcwd()
    t1 = True
    
    # Now start the conversion of the dataset to nnU-Net:
    if t1:
        task_name = 'Task04_MWSC-t1'
    else:
        task_name = 'Task03_MWSC'
    target_base = join(nnFormer_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")

    # Remember to make the actual directories
    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTr)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTs)
    
    # test set
    if t1:
        images_dir_ts = join(base, '../nobackup/MWSC_FlairT1')
    else:
        images_dir_ts = join(base, '../nobackup/MWSC_challenge/mwsc_testset', 'image_masked')
    testing_images = subfiles(images_dir_ts, join=False)
    for t in tqdm(testing_images):
        input_image_file = join(images_dir_ts, t)
        unique_name = t[:7]
        output_image_file = join(target_imagesTs, unique_name + ".nii.gz")

        shutil.copy(input_image_file, output_image_file)
        shutil.copy(input_image_file, join(target_imagesTr, unique_name + ".nii.gz"))

    labels_dir_ts = join(base, '../nobackup/MWSC_challenge/mwsc_testset', 'label_masked')
    test_labels = subfiles(labels_dir_ts, join=False)
    for t in tqdm(test_labels):
        input_seg_file = join(labels_dir_ts, t)
        unique_name = t[:7]
        output_seg_file = join(target_labelsTs, unique_name + ".nii.gz")

        shutil.copy(input_seg_file, output_seg_file)
        shutil.copy(input_seg_file, join(target_labelsTr, unique_name + ".nii.gz"))

    # finally we can call a utility function for generating a dataset.json
    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, modalities=('FLAIR',), 
                          labels={0: 'background', 1: 'wmh'}, dataset_name=task_name, license='MICCAI')
