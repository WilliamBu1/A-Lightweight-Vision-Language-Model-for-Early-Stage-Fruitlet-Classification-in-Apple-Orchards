# TinyCLIP Fine-tuning and Heatmap Pipeline for Apple Fruitlet Localization

## Overview
Two-stage approach for localizing apple fruitlet anatomical structures in high-resolution orchard images using a fine-tuned TinyCLIP vision-language model.

## Notebooks

### 1. TinyCLIP Fine-tuning (`tinyCLIP.ipynb`)
**Purpose**: Fine-tune TinyCLIP model on small image patches for multi-label classification

**Model**: TinyCLIP-ViT-61M-32-Text-29M-LAION400M
- Lightweight vision-language model optimized for edge deployment
- The finnetuned weights are in this repo but you can finetune it yourself using the dataset and notebook

**Dataset**: 
- 2,726 apple fruitlet image patches (224x224)
- 4 classes: calyx, fruitlet, peduncle, negative
- Apple varieties: Scilate and Scifresh orchards

**Training**:
- 5 epochs, batch size 32, BCEWithLogitsLoss
- Text prompts: "a photo of a {class_name}"

**Evaluation**:
- Classification metrics, confusion matrices, precision-recall curves
- Performance benchmarking (model size, inference speed, memory usage)
- Misclassification analysis

### 2. Heatmap Pipeline (`pipeline.ipynb`)
**Purpose**: Apply fine-tuned model to full high-resolution orchard images using sliding window analysis

**Process**:
1. **Sliding Window**: Extract overlapping 224x224 patches from full images (stride=112)
2. **Batch Processing**: Process patches in batches of 64 for memory efficiency
3. **Heatmap Generation**: Aggregate patch predictions into spatial probability maps
4. **Visualization**: Generate class-specific heatmaps for anatomical structure localization

**Input**: High-resolution orchard images with COCO annotations
**Output**: Visual heatmaps showing predicted locations of fruitlet structures

## Architecture Overview
- **Patch-to-Image Pipeline**: Model trained on small patches generalizes to localize structures in full-resolution images
- **Efficient Processing**: Batch processing prevents memory overflow on hundreds of patches per image
- **Visual Localization**: Converts patch-level classifications into interpretable spatial heatmaps

## Limitations
- Pipeline provides visual localization without quantitative evaluation metrics
- Limited by dataset quality for comprehensive performance assessment
- Evaluation focuses on qualitative heatmap analysis rather than precise localization accuracy

## Applications
- Automated apple quality assessment
- Precision agriculture monitoring  
- Robotic harvesting guidance
- Orchard management decision support
