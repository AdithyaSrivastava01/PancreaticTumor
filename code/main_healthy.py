from slicing import find_slice

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import math
import cv2 as cv

import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

import os

from nifti2dicom import convertNsave


path = "../Segment_data/Healthy_pancreas/Pancreas_slice/"

dcm_mask_path = "../Segment_data/Healthy_pancreas/Mask"
dcm_result_path = "../Segment_data/Healthy_pancreas/Result"

for i in os.listdir(path):
    x=i.split('_')[2].split('.')[0]
    path_new="../dataset/Healthy_Pancreas/TCIA_pancreas_labels-02-05-2017/"+"label"+f"{x}.nii.gz"
    test_load = pydicom.dcmread(path+i).pixel_array

    test_load_label=nib.load(path_new).get_fdata()
    slices = test_load_label.shape[2]
    pancreas_best = find_slice(test_load_label,slices,1)
    test_load_label[:, :, pancreas_best] = np.rot90(test_load_label[:, :, pancreas_best],k=1)
    test_load_label[:, :, pancreas_best] = np.flip(test_load_label[:, :, pancreas_best],axis=0)
    result = test_load * test_load_label[:,:,pancreas_best]
    result[result == 0] = -1024.

    y = i.split('.')[0]

    dicom_dir_label = os.path.join(dcm_mask_path, f"{y}.dcm")
    dicom_dir_result = os.path.join(dcm_result_path, f"{y}.dcm")

    convertNsave(test_load_label[:, :, pancreas_best],dicom_dir_label,1)
    convertNsave(result,dicom_dir_result,0)

