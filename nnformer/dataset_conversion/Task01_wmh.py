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
    task_name = 'Task01_wmh'
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

    # Convert the training examples
    images_dir_tr = join(base, '../nobackup/VoSHT/data', 'Training', 'images')

    # Locate all images
    training_images = subfiles(images_dir_tr, join=False)
    for t in tqdm(training_images):
        input_image_file = join(images_dir_tr, t)
        #unique_name = t[:-7]  # just the filename with the .nii.gz extension cropped away
        unique_name = t[:18]
        output_image_file = join(target_imagesTr, unique_name + ".nii.gz")

        shutil.copy(input_image_file, output_image_file)

    labels_dir_tr = join(base, '../nobackup/VoSHT/data', 'Training', 'labels')
    training_labels = subfiles(labels_dir_tr, join=False)
    for t in tqdm(training_labels):
        input_seg_file = join(labels_dir_tr, t)
        #unique_name = t
        unique_name = t[:18]
        output_seg_file = join(target_labelsTr, unique_name + ".nii.gz")

        shutil.copy(input_seg_file, output_seg_file)
    
     # now do the same for the test set
    images_dir_ts = join(base, '../nobackup/VoSHT/data', 'Test', 'images')
    testing_images = subfiles(images_dir_ts, join=False)
    for t in tqdm(testing_images):
        input_image_file = join(images_dir_ts, t)
        #unique_name = t[:-7]
        unique_name = t[:18]
        output_image_file = join(target_imagesTs, unique_name + ".nii.gz")

        shutil.copy(input_image_file, output_image_file)

    labels_dir_ts = join(base, '../nobackup/VoSHT/data', 'Test', 'labels')
    test_labels = subfiles(labels_dir_ts, join=False)
    for t in tqdm(test_labels):
        input_seg_file = join(labels_dir_ts, t)
        #unique_name = t
        unique_name = t[:18]
        output_seg_file = join(target_labelsTs, unique_name + ".nii.gz")

        shutil.copy(input_seg_file, output_seg_file)

    # finally we can call the utility for generating a dataset.json
    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, modalities=('FLAIR',), 
                          labels={0: 'background', 1: 'wmh'}, dataset_name=task_name, license='DRCMR')

    # generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, modalities=('T1', 'T1ce', 'T2', 'FLAIR'),
    #                       labels={"0": "background", "1": "edema", "2": "non-enhancing", "3": "enhancing"}, dataset_name=task_name, license='Test_BRATS')