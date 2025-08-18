# TinyCLIP Quantization with ONNX → TensorRT Engine and Heatmap Pipeline for Apple Fruitlet Localization  

## Overview  
This section of the repository describes the deployment of a fine-tuned **TinyCLIP** model on the NVIDIA Jetson Orin platform. The pipeline focuses on quantization and model optimization for real-time edge inference, followed by a TensorRT-accelerated heatmap generation process for localizing apple fruitlet anatomical structures.  

> ⚠️ **Note:** Some paths in the scripts and notebooks may need to be adjusted based on your local setup. Additionally, the Python environment is sensitive due to dependency conflicts — careful setup is recommended.

---

## Workflow  

### 1. Model Conversion  
1. **Export to ONNX**  
   - Fine-tuned TinyCLIP PyTorch model exported to ONNX format (`placeholder_model.onnx`).  
   - Verified for dynamic input shape compatibility with 224×224 patches.  

2. **Quantization & TensorRT Engine Build**  
   - Convert ONNX to TensorRT using the provided script: `onnx2trt.py`.  
   - Supports FP16 and INT8 quantization.  
   - Produces TensorRT engine files (`placeholder_model_fp16.trt`, `placeholder_model_int8.trt`) optimized for **batch size = 8** on Jetson Orin.  

---

### 2. Heatmap Pipeline (Jetson-Optimized)  
1. **Sliding Window Extraction**: Generate overlapping patches (224×224, stride=112) from high-resolution orchard images.  
2. **TensorRT Inference**: Perform classification on batches of patches using the TensorRT engine.  
3. **Heatmap Aggregation**: Assemble predictions into spatial probability maps.  
4. **Visualization**: Generate class-specific heatmaps highlighting calyx, fruitlet, and peduncle regions.  

---

## Python Scripts  

- **`onnx2trt.py`**  
  - Converts an ONNX model into a TensorRT engine (`.trt`) with optional FP16 or INT8 quantization.  
  - Batch size and workspace memory can be adjusted inside the script.  

- **`tinyCLIPeval.py` / `final_eval.py` (Combined Evaluation)**  
  - Perform image patch evaluation using TensorRT engines.  
  - Computes classification metrics for each patch and aggregates results across datasets.  
  - Outputs structured results for further analysis or visualization.  

> ⚠️ **Warning:** Users may need to modify file paths in the scripts for input images, ONNX models, or TRT engines.  

---

## Notebooks  

- **`sliding_window.ipynb`**  
  - Demonstrates the heatmap generation pipeline.  
  - Uses sliding window extraction, TensorRT inference, and patch aggregation to produce class-specific heatmaps of apple fruitlet structures.  

> ⚠️ **Warning:** The notebook may require path adjustments depending on your local environment and folder structure.  

---

## Python Environment  

- **Requirements**: `requirements.txt`  
  - Contains all dependencies for Jetson inference and heatmap generation.  
  - ⚠️ **Caution:** Some packages may conflict or fail to install due to version incompatibilities. It is recommended to use a dedicated Python virtual environment or conda environment.  

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
- INT8 quantization may fall back to FP16 in unsupported layers.  
- Heatmaps remain qualitative; quantitative localization metrics are not included.  
- Python environment is sensitive; users may need to carefully manage dependencies.  

