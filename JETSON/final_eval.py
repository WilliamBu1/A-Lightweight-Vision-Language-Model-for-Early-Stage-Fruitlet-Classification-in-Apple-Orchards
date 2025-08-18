import torch
from transformers import CLIPProcessor
from PIL import Image
import os
import numpy as np
from tqdm.auto import tqdm
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# ===================================================================
# 1. CONFIGURATION
# ===================================================================
# --- Path to your pre-computed text features and logit scale ---
ASSETS_PATH = './clip_assets.pt'

# --- Your final, complete TensorRT engine ---
TRT_ENGINE_PATH = './tinyclip_int8_dynamic.trt'  # Or './tinyclip_int8.trt'

# --- Your validation dataset ---
VALIDATION_DIR = './valid_crops'

# --- Model and Class Configuration ---
MODEL_ID = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"
CLASS_LABELS = ["calyx", "fruitlet", "peduncle", "negative"]
# ===================================================================

def allocate_buffers(engine, context):
    """Allocate buffers for TensorRT engine using modern API."""
    inputs = []
    outputs = []
    bindings = []
    
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_shape = context.get_tensor_shape(tensor_name)
        tensor_dtype = engine.get_tensor_dtype(tensor_name)
        
        # Calculate size correctly
        size = trt.volume(tensor_shape)
        dtype = trt.nptype(tensor_dtype)
        
        print(f"Tensor: {tensor_name}, Shape: {tensor_shape}, Size: {size}, DType: {dtype}")
        
        # Allocate host and device memory
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        # Store binding information
        bindings.append(int(device_mem))
        
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append({
                'host': host_mem, 
                'device': device_mem,
                'name': tensor_name,
                'shape': tensor_shape,
                'dtype': dtype
            })
        else:
            outputs.append({
                'host': host_mem, 
                'device': device_mem,
                'name': tensor_name,
                'shape': tensor_shape,
                'dtype': dtype
            })
    
    return inputs, outputs, bindings

def get_cuda_memory_usage():
    """Get the total used memory in the current CUDA context."""
    try:
        free_mem, total_mem = cuda.mem_get_info()
        used_mem = (total_mem - free_mem) / (1024 * 1024)  # Convert to MB
        return used_mem
    except Exception as e:
        print(f"Error querying CUDA memory: {e}")
        return 0

def main():
    """Main function to run the full evaluation."""
    if not torch.cuda.is_available():
        print("❌ Error: A CUDA-enabled GPU is required.")
        return

    device = torch.device("cuda")

    # ===================================================================
    # 2. LOAD ASSETS AND TENSORRT ENGINE
    # ===================================================================
    print("--- 1. Loading Assets & TRT Engine ---")

    # --- Load pre-computed assets (text features and logit scale) ---
    print(f"Loading pre-computed assets from: {ASSETS_PATH}")
    if not os.path.exists(ASSETS_PATH):
        raise FileNotFoundError(f"Assets file not found at {ASSETS_PATH}. Please run the 'create_deployment_assets.py' script first.")
    
    clip_assets = torch.load(ASSETS_PATH)
    text_features = clip_assets['text_features'].to(device)
    logit_scale = clip_assets['logit_scale'].to(device)
    print("✅ Text features and logit scale loaded.")

    # --- Load the universal processor (for images only) ---
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    # --- Load TensorRT Engine and Allocate Buffers ---
    print(f"Loading TensorRT engine from: {TRT_ENGINE_PATH}")
    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)
    with open(TRT_ENGINE_PATH, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # --- Set input shape for dynamic engine (assuming batch size 1) ---
    input_shape = (1, 3, 224, 224)  # Typical CLIP input shape
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            context.set_input_shape(tensor_name, input_shape)
            break

    # --- Allocate memory for inputs and outputs using modern API ---
    try:
        inputs, outputs, bindings = allocate_buffers(engine, context)
        stream = cuda.Stream()
        print("✅ TensorRT engine loaded and buffers allocated.")
    except cuda.MemoryError as e:
        print(f"❌ Memory allocation failed: {e}")
        print("💡 Try reducing batch size or using a model with smaller memory requirements")
        return

    # --- Collect image paths ---
    image_paths = []
    for root, _, files in os.walk(VALIDATION_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    print(f"Found {len(image_paths)} images for evaluation.")

    # ===================================================================
    # 3. RUN EVALUATION AND MEASUREMENTS
    # ===================================================================
    print("\n--- 2. Running Evaluation & Benchmarking ---")

    all_true_labels = []
    all_predicted_labels = []
    latencies = []
    peak_memory_cuda = 0
    
    # --- Reset memory stats before the loop ---
    torch.cuda.reset_peak_memory_stats(device)
    baseline_memory = get_cuda_memory_usage()
    print(f"Baseline GPU Memory (CUDA context): {baseline_memory:.2f} MB")

    # --- GPU Warm-up ---
    print("Warming up GPU...")
    for _ in range(10):
        for inp in inputs:
            context.set_tensor_address(inp['name'], inp['device'])
        for out in outputs:
            context.set_tensor_address(out['name'], out['device'])
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
    
    print("Starting evaluation...")
    pbar = tqdm(image_paths, desc="Evaluating with TensorRT")
    for image_path in pbar:
        true_label = os.path.basename(os.path.dirname(image_path))
        image = Image.open(image_path).convert("RGB")
        
        # --- Preprocess image ---
        image_input = processor(images=image, return_tensors="pt").to(device)
        pixel_values = image_input['pixel_values']

        # --- Latency Measurement ---
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # --- TensorRT Inference using modern API ---
        np.copyto(inputs[0]['host'], pixel_values.cpu().numpy().ravel())
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
        
        for inp in inputs:
            context.set_tensor_address(inp['name'], inp['device'])
        for out in outputs:
            context.set_tensor_address(out['name'], out['device'])
        
        context.execute_async_v3(stream_handle=stream.handle)
        
        for out in outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
        
        stream.synchronize()

        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))

        # --- Track peak GPU memory ---
        current_memory = get_cuda_memory_usage()
        peak_memory_cuda = max(peak_memory_cuda, current_memory)

        # --- Post-process output ---
        raw_output = torch.from_numpy(outputs[0]['host']).to(device)
        expected_feature_dim = text_features.shape[-1]
        image_features = raw_output.reshape(1, expected_feature_dim) 
        
        with torch.no_grad():
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            logits_per_image = logit_scale * image_features @ text_features.t()
            predicted_index = logits_per_image.argmax().item()
            
        all_predicted_labels.append(CLASS_LABELS[predicted_index])
        all_true_labels.append(true_label)

    # ===================================================================
    # 4. REPORT RESULTS
    # ===================================================================
    print("\n\n--- 3. Final Report ---")
    
    # --- Performance Metrics ---
    engine_size_mb = os.path.getsize(TRT_ENGINE_PATH) / (1024 * 1024)
    avg_latency_ms = np.mean(latencies)
    throughput_fps = 1000.0 / avg_latency_ms
    peak_memory_pytorch = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    
    print("\n--- Performance Metrics ---")
    print(f"✅ Engine File Size: {engine_size_mb:.2f} MB")
    print(f"✅ Average Latency: {avg_latency_ms:.2f} ms")
    print(f"✅ Throughput: {throughput_fps:.2f} FPS (images/sec)")
    print(f"✅ Peak GPU Memory (CUDA context): {peak_memory_cuda:.2f} MB")
    print(f"✅ Peak GPU Memory (PyTorch components): {peak_memory_pytorch:.2f} MB")

    # --- Accuracy Metrics ---
    if all_true_labels:
        print("\n--- Accuracy Metrics ---")
        report = classification_report(all_true_labels, all_predicted_labels, labels=CLASS_LABELS, digits=4)
        print(report)

        # --- Confusion Matrix ---
        cm = confusion_matrix(all_true_labels, all_predicted_labels, labels=CLASS_LABELS)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
        plt.title('TensorRT Engine Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    # Cleanup
    del inputs, outputs, bindings, context, engine

if __name__ == "__main__":
    print("🚀 To ensure accurate benchmarks, please run 'sudo jetson_clocks' in your terminal first.")
    main()