# TinyCLIP Fine-Tuning and Heatmap Pipeline for Apple Fruitlet Localization  

## Overview  
This repository presents a two-stage approach for localizing apple fruitlet anatomical structures in high-resolution orchard images using a fine-tuned **TinyCLIP** vision-language model.  

⚠️ **Note:** Paths in the notebooks may require adjustment depending on your specific setup. In most cases, alignment should be straightforward if the necessary data is uploaded to the Google Colab session.  

---

## Python Scripts  
The dataset preprocessing pipeline generates targeted structural crops from the original full-resolution images using the `train_list1_2.py` script. This process automatically creates a hierarchical directory structure (`crops/`) with four subdirectories corresponding to anatomical and structural regions of interest. The preprocessing operates on both the training (`main/train/`) and validation (`main/valid/`) partitions.  

Pre-processed cropped images are included in the repository to facilitate immediate experimentation. Users may optionally re-execute the preprocessing pipeline to modify crop parameters or regenerate the dataset with adjusted extraction criteria.  

---

## Dataset  
The dataset consists of high-resolution orchard images and derived cropped patches:  

- **Source**: 600 full-sized images from Roboflow with COCO annotations  
- **Split**: 80/10/10 for training, validation, and testing  
- **Derived Data**: Thousands of cropped apple fruitlet patches (224×224)  
- **Classes**: Calyx, Fruitlet, Peduncle, Negative  
- **Varieties**: Scilate and Scifresh orchards  

*Note: Training and validation sets are provided; the test set is not included.*  

---

## Notebooks  

### 1. TinyCLIP Fine-Tuning (`tinyCLIP.ipynb`)  
**Purpose**: Fine-tune TinyCLIP on cropped image patches for multi-label classification.  

**Model**: `TinyCLIP-ViT-61M-32-Text-29M-LAION400M`  
- Lightweight vision-language model optimized for edge deployment  
- Finetuned weights are included, but users may re-train using the dataset and notebook  

**Training Configuration**:  
- 5 epochs, batch size 32  
- Loss: `BCEWithLogitsLoss`  
- Text prompts: *“a photo of a {class_name}”*  

**Evaluation Metrics**:  
- Classification performance (accuracy, precision, recall, F1)  
- Confusion matrices and precision-recall curves  
- Resource benchmarking (model size, inference speed, memory usage)  
- Misclassification analysis  

---

### 2. Heatmap Pipeline (`pipeline.ipynb`)  
**Purpose**: Apply the fine-tuned model to full-resolution orchard images using a sliding window approach.  

**Pipeline Steps**:  
1. **Sliding Window Extraction**: Generate overlapping 224×224 patches (stride=112).  
2. **Batch Inference**: Process patches in batches of 64 to optimize memory usage.  
3. **Heatmap Aggregation**: Convert patch-level predictions into spatial probability maps.  
4. **Visualization**: Produce class-specific heatmaps indicating anatomical structures.  

**Input**: High-resolution orchard images with COCO annotations  
**Output**: Heatmaps highlighting predicted locations of fruitlet structures  

---

## Architecture Overview  
- **Patch-to-Image Generalization**: Model trained on small patches extends to structure localization in full-resolution images.  
- **Efficient Processing**: Batch inference enables scaling to hundreds of patches per image without memory overflow.  
- **Interpretable Outputs**: Patch-level classifications are aggregated into spatial heatmaps for intuitive visualization.  

---

## Limitations  
- Localization is visual and qualitative; quantitative evaluation metrics are not included.  
- Performance is constrained by dataset quality and annotation consistency.  
- The pipeline emphasizes interpretability through heatmaps rather than precise localization accuracy.  

