# Imports
from batchgenerators.utilities.file_and_folder_operations import *
from utils import generate_dataset_json
from nnformer.paths import nnFormer_raw_data
from tqdm import tqdm
import os
import shutil

if __name__ == '__main__': 
    base = os.getcwd()

    # Now start the conversion of the dataset to nnU-Net:
    task_name = 'Task02_t1'
    target_base = join(nnFormer_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")
    target_labelsTr = join(target_base, "labelsTr")

    # Remember to make the actual directories
    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)


    # Go through all labels first to get the names in split to replicate that for the images
    
    training_names = []
    labels_dir_tr = join(base, '../nobackup/VoSHT/data', 'Training', 'labels')
    training_labels = subfiles(labels_dir_tr, join=False)
    
    for t in tqdm(training_labels):
        input_seg_file = join(labels_dir_tr, t)
        unique_name = t[:18]
        training_names.append(unique_name)
        output_seg_file = join(target_labelsTr, unique_name + ".nii.gz")

        shutil.copy(input_seg_file, output_seg_file)
    
    test_names = []
    labels_dir_ts = join(base, '../nobackup/VoSHT/data', 'Test', 'labels')
    test_labels = subfiles(labels_dir_ts, join=False)

    for t in tqdm(test_labels):
        input_seg_file = join(labels_dir_ts, t)
        unique_name = t[:18]
        test_names.append(unique_name)
        output_seg_file = join(target_labelsTs, unique_name + ".nii.gz")

        shutil.copy(input_seg_file, output_seg_file)


    # Now go through the 4d images and split them accordingly into train and test
    images_dir = join(base, '../nobackup/VoSHT/data', 'FlairT1')
    images = subfiles(images_dir, join=False)

    for i in tqdm(images):
        input_image_file = join(images_dir, i)
        unique_name = i[:18]

        if unique_name in training_names:
            output_image_file = join(target_imagesTr, unique_name + ".nii.gz")
        elif unique_name in test_names:
            output_image_file = join(target_imagesTs, unique_name + ".nii.gz")
        else:
            raise IndexError('Image is not a train or test images!')

        shutil.copy(input_image_file, output_image_file)
    

    # Finally we can call a utility function for generating a dataset.json
    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, modalities=('FLAIR', 'T1'), 
                          labels={0: 'background', 1: 'wmh'}, dataset_name=task_name, license='DRCMR')
