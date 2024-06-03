from slicing import find_slice

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import math
import cv2 as cv

import pydicom
import os

from nifti2dicom import convertNsave

path = "/home/adithya/Document/Major_Proj/Segmentation_MRCNN/Pancreas_slice"

dcm_slice_path  = "../Segment_data/Tumor_pancreas/Pancreas_slice"
dcm_mask_path = "../Segment_data/Tumor_pancreas/Mask"
dcm_result_path = "../Segment_data/Tumor_pancreas/Result"

for i in os.listdir(path):
    x=i.split('.')[0]
    path_label = '/home/adithya/Document/Major_Proj/Segmentation_MRCNN/Mask/'+f'{x}.dcm'
    path_slice = "/home/adithya/Document/Major_Proj/Segmentation_MRCNN/Pancreas_slice/"+f'{x}.dcm'

    test_load = pydicom.dcmread(path_slice).pixel_array
    test_load_label = pydicom.dcmread(path_label).pixel_array

    # slices=test_load.shape[2]

    # pancreas_best = find_slice(test_load_label,slices,1)
    # tumor_best = find_slice(test_load_label,slices,2)
    


    new_label = np.maximum(test_load_label,test_load_label)




    threshold=0.
    ret,binary_mask=cv.threshold(new_label,threshold,2.,cv.THRESH_BINARY)
    binary_mask[binary_mask != 0] = 1.

    result = binary_mask*test_load
    result[result == 0] = -1024.

    x=os.path.split(path_slice)[-1].split(".")[0]

    dicom_dir_slice = os.path.join(dcm_slice_path, f"{x}.dcm")
    dicom_dir_label = os.path.join(dcm_mask_path, f"{x}.dcm")
    dicom_dir_result = os.path.join(dcm_result_path, f"{x}.dcm")

    # convertNsave(test_load,dicom_dir_slice,0)
    # convertNsave(new_label,dicom_dir_label,1)
    convertNsave(result,dicom_dir_result,0)












