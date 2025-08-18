# TinyCLIP Quantization with ONNX → TensorRT Engine and Heatmap Pipeline for Apple Fruitlet Localization  

## Overview  
This section of the repository describes the deployment of a fine-tuned **TinyCLIP** model on the NVIDIA Jetson Orin platform. The pipeline focuses on quantization and model optimization for real-time edge inference, followed by a TensorRT-accelerated heatmap generation process for localizing apple fruitlet anatomical structures.  

---

## Workflow  

### 1. Model Conversion  
1. **Export to ONNX**  
   - Fine-tuned TinyCLIP PyTorch model exported to ONNX format (`.onnx`).  
   - Verified for dynamic input shape compatibility with 224×224 patches.  

2. **Quantization**  
   - FP16 and INT8 quantization supported.  
   - INT8 calibration performed using a representative dataset of apple fruitlet patches.  

3. **TensorRT Engine Build**  
   - ONNX model converted into a TensorRT engine (`.trt`) optimized for Jetson Orin.  
   - Engine tuned for batch processing of patches in the range 1–32; optimized for **batch size = 8**.  

---

### 2. Heatmap Pipeline (Jetson-Optimized)  
1. **Sliding Window Extraction**: Generate overlapping patches (224×224, stride=112) from high-resolution orchard images.  
2. **TensorRT Inference**: Perform classification on batches of patches using the TensorRT engine.  
3. **Heatmap Aggregation**: Assemble predictions into spatial probability maps.  
4. **Visualization**: Generate class-specific heatmaps highlighting calyx, fruitlet, and peduncle regions.  

---

## Python Scripts  

- **`tinyCLIPeval.py` / `final_eval.py` (Combined Evaluation)**  
  - Perform image patch evaluation using TensorRT engines.  
  - Computes classification metrics for each patch and aggregates results across datasets.  
  - Outputs structured results for further analysis or visualization.  

---

## Notebooks  

- **`sliding_window.ipynb`**  
  - Demonstrates the heatmap generation pipeline.  
  - Uses sliding window extraction, TensorRT inference, and patch aggregation to produce class-specific heatmaps of apple fruitlet structures.  

---

## Desired Performance  
- **Inference Speed**: Real-time inference achieved on Jetson Orin with FP16 quantization.  
- **Resource Efficiency**: TensorRT engine significantly reduces GPU memory usage compared to PyTorch inference.  
- **Scalability**: Capable of processing hundreds of patches per full-resolution image without exceeding Orin memory limits.  

---

## Requirements  
- **Hardware**: NVIDIA Jetson Orin (8GB / 16GB recommended)  
- **Software Stack**:  
  - JetPack SDK (>= 5.0)  
  - TensorRT (>= 8.x)  
  - PyTorch / Torch-TensorRT (for conversion)  
  - OpenCV, NumPy, Matplotlib (for preprocessing & visualization)  

---

## Limitations  
- INT8 quantization failed in many layers and fell back to FP16.  
- Heatmaps remain qualitative; quantitative localization metrics are not included.  
