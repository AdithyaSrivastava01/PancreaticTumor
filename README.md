# Improvised Pancreatic Tumor Detection using CT Radiomics Predictions  

## Overview  

Pancreatic cancer is one of the most lethal malignancies, with poor survival rates due to late-stage detection. Current diagnostic methods like CT and MRI scans are costly and may yield false negatives, leading to delayed treatment. This research introduces an advanced, fully automated system leveraging deep learning and radiomics to enhance pancreatic tumor detection in CT scans.  

### Key Contributions:  
- **Deep Instance Segmentation**: Utilized **Mask R-CNN** to accurately segment pancreatic tumor regions.  
- **Radiomics-Based Feature Extraction**: Extracted computationally rich features such as texture, shape, and intensity from segmented tumor regions.  
- **Feature Fusion**: Combined extracted features from both pancreas and tumor regions in an optimal **70:30 ratio**.  
- **Classification**: Employed multiple ML classifiers, with **AdaBoost achieving state-of-the-art accuracy of 93.3%**.  

## Methodology  

### 1. **Segmentation using Mask R-CNN**  
- Applied **ResNet-50** as the backbone for **Mask R-CNN** to accurately delineate tumor boundaries.  
- Achieved **87.29% Dice Similarity Coefficient (DSC)**, surpassing U-Net-based segmentation methods.  

### 2. **Feature Extraction with Radiomics**  
- Extracted **102 radiomic features**, including:  
  - **Shape-Based Features** (e.g., elongation, compactness)  
  - **Texture Analysis** (GLCM, GLRLM, NGTDM matrices)  
  - **First-Order Statistical Features** (e.g., entropy, mean intensity)  

### 3. **Feature Fusion**  
- Integrated tumor and pancreatic region features in an optimal **70:30 ratio**, significantly improving classification accuracy.  

### 4. **Classification using Machine Learning Models**  
- **AdaBoost performed best with 93.3% accuracy**, followed by XGBoost (91.25%) and a Feedforward Neural Network (88.4%).  


## Results  

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |  
|--------|----------|------------|--------|---------|---------|  
| **AdaBoost** | **93.3%** | 96.12% | 96.78% | 96.40% | 97.72% |  
| XGBoost | 91.25% | 92.1% | 91.3% | 94.3% | 92% |  
| Feedforward Neural Network | 88.4% | 89% | 90.1% | 87.8% | 91% |  

### **Segmentation Performance (Mask R-CNN vs. U-Net)**  
| Model | Dice Coefficient | IoU | Pixel Accuracy |  
|--------|----------------|------|----------------|  
| **Mask R-CNN** | **87.29%** | **82.10%** | **98.83%** |  
| U-Net | 51.97% | 46.97% | 49.49% |  

## Dataset  
- **Healthy pancreas scans**: 80 patients from *The Cancer Imaging Archive (TCIA)*.  
- **Tumor-confirmed scans**: 280 patients from *Memorial Sloan Kettering Cancer Center (MSKCC)*.  
- **Total CT images**: 6325, augmented to **1440 images** via flipping and rotation techniques.  

## Experimental Setup  
- **Mask R-CNN** trained for **22 epochs** using **Adam optimizer** (learning rate: 0.0001).  
- **FNN** trained for **24 epochs**, incorporating **learning rate scheduler** and **dropout (0.2)** to prevent overfitting.  
- **AdaBoost** trained with **50 weak learners**, outperforming other classifiers.  

## Conclusion  
This research presents a novel **multi-stage deep learning approach** combining segmentation, radiomics feature extraction, and machine learning classifiers to improve pancreatic tumor detection accuracy. With an **accuracy of 93.3%**, this method outperforms existing state-of-the-art approaches. Future work will explore **3D CT image analysis** and **cross-database generalization** to enhance diagnostic performance further.  

## References  
- [1] The Cancer Imaging Archive: [Link](https://wiki.cancerimagingarchive.net/display/public/pancreas-ct)  
- [2] Memorial Sloan Kettering Cancer Center: [Link](http://medicaldecathlon.com/)  
- Additional references and more details are available in the full research paper(under review): [Link](https://drive.google.com/drive/folders/1OZpMVjTcZ3_rYUhTuIra5YogLBDw6kCg).  
