import pydicom

def convertNsave(arr,path,category):
    
    
    dicom_file = pydicom.dcmread('../Segment_data/Tumor_pancreas/Pancreas_slice/pancreas_001.dcm')
    arr = arr.astype('int16')
    dicom_file.Rows = arr.shape[0]
    dicom_file.Columns = arr.shape[1]
    dicom_file.PhotometricInterpretation = "MONOCHROME2"
    dicom_file.SamplesPerPixel = 1
    dicom_file.BitsStored = 16
    dicom_file.BitsAllocated = 16
    dicom_file.HighBit = 15

    if category==0:
        dicom_file.PixelRepresentation = 1
        dicom_file.PixelData = arr.tobytes()
        dicom_file.save_as(path)

    else: 
        dicom_file.PixelRepresentation = 0
        dicom_file.PixelData = arr.tobytes()
        dicom_file.save_as(path)

    
