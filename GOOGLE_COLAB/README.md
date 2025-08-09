# TinyCLIP Fine-tuning for Apple Fruitlet Detection

## Overview
Fine-tuning TinyCLIP model to classify apple fruitlet structures in orchard images.

## Model
- **TinyCLIP-ViT-61M-32-Text-29M-LAION400M**
- Lightweight vision-language model

## Dataset
- **2,726 apple fruitlet image patches**
- **4 classes**: calyx, fruitlet, peduncle, negative
- **Varieties**: Scilate and Scifresh orchards

## Training
- 5 epochs, batch size 32
- Multi-label classification with BCEWithLogitsLoss
- Text prompts: "a photo of a {class_name}"

## Evaluation
- Classification metrics and confusion matrices
- Precision-recall curves
- Performance benchmarking (model size, inference speed, memory usage)
- Misclassification analysis

## Applications
- Automated apple quality assessment
- Precision agriculture monitoring
- Robotic harvesting guidance
