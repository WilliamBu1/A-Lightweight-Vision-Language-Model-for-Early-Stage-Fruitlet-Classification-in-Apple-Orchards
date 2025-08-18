import tensorrt as trt
import os
import torch
from PIL import Image
import numpy as np
import glob
import pycuda.driver as cuda
import pycuda.autoinit
from transformers import CLIPProcessor

# ===================================================================
# 1. CONFIGURATION
# ===================================================================

# --- Folder containing ~100-500 sample images for INT8 calibration ---
CALIBRATION_IMAGE_DIR = 'calibration_data/'

# --- Path to the ONNX file created by your corrected export script ---
ONNX_MODEL_PATH = "tinyclip_dynamic.onnx"

# --- Desired output paths for the new engines ---
TRT_FP16_PATH = "tinyclip_fp16_dynamic.trt"
TRT_INT8_PATH = "tinyclip_int8_dynamic.trt"

# --- Model ID (used for loading the correct processor) ---
MODEL_ID = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"

# --- Dynamic batch size configuration ---
MIN_BATCH_SIZE = 1
OPT_BATCH_SIZE = 8  # Optimal batch size for performance
MAX_BATCH_SIZE = 32  # Maximum batch size you expect to use

# ===================================================================

class Int8Calibrator(trt.IInt8EntropyCalibrator):
    """
    A robust INT8 Calibrator that uses the Hugging Face CLIPProcessor
    to ensure perfect preprocessing alignment with dynamic batch support.
    """
    def __init__(self, image_dir, cache_file="int8_calibration.cache"):
        super().__init__()
        print("Initializing Calibrator...")
        # Use optimal batch size for calibration
        self.batch_size = OPT_BATCH_SIZE
        self.cache_file = cache_file

        # Find all images in the calibration directory
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not self.image_files:
            raise ValueError(f"No calibration images found in {image_dir}")

        self.num_images = len(self.image_files)
        self.current_index = 0

        # Load the processor to ensure preprocessing is identical
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)
        
        # Allocate device memory for a single batch
        self.device_input = cuda.mem_alloc(self.batch_size * 3 * 224 * 224 * np.dtype(np.float32).itemsize)
        print(f"Found {self.num_images} images. Calibrating with batch size {self.batch_size}.")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= self.num_images:
            return None  # No more batches

        # Get the next batch of file paths
        batch_end = min(self.current_index + self.batch_size, self.num_images)
        batch_files = self.image_files[self.current_index:batch_end]
        self.current_index = batch_end
        
        # Load and preprocess images
        pil_images = [Image.open(f).convert('RGB') for f in batch_files]
        
        # Use the correct processor to generate a numpy array
        inputs = self.processor(images=pil_images, return_tensors="np")
        batch_data = inputs['pixel_values']

        # Pad the batch if it's the last one and smaller than batch_size
        if batch_data.shape[0] < self.batch_size:
            padded_batch = np.zeros((self.batch_size, 3, 224, 224), dtype=np.float32)
            padded_batch[:batch_data.shape[0]] = batch_data
            batch_data = padded_batch

        # Copy data to the GPU
        cuda.memcpy_htod(self.device_input, batch_data)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"INFO: Reading calibration cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print(f"INFO: Writing calibration cache to {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def setup_dynamic_shapes(network, config, builder):
    """
    Configure dynamic shapes and optimization profiles for the network.
    This is crucial for enabling dynamic batch sizes.
    Compatible with different TensorRT versions.
    """
    # Find the input tensor (assuming it's the first input)
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    
    print(f"Setting up dynamic shapes for input: {input_name}")
    print(f"Input shape: {input_tensor.shape}")
    
    # Define the dynamic shapes: (batch, channels, height, width)
    # Assuming input shape is [batch, 3, 224, 224]
    min_shape = (MIN_BATCH_SIZE, 3, 224, 224)
    opt_shape = (OPT_BATCH_SIZE, 3, 224, 224)  
    max_shape = (MAX_BATCH_SIZE, 3, 224, 224)
    
    # Try different API versions in order of preference
    profile = None
    
    # Method 1: Try builder.create_optimization_profile() (most common)
    try:
        profile = builder.create_optimization_profile()
        print("✅ Created optimization profile via builder")
    except (AttributeError, TypeError):
        pass
    
    # Method 2: Try config.create_optimization_profile()
    if profile is None:
        try:
            profile = config.create_optimization_profile()
            print("✅ Created optimization profile via config")
        except (AttributeError, TypeError):
            pass
    
    # Method 3: Try trt.Builder.create_optimization_profile()
    if profile is None:
        try:
            profile = trt.Builder.create_optimization_profile(builder)
            print("✅ Created optimization profile via static method")
        except (AttributeError, TypeError):
            pass
    
    if profile is None:
        print("❌ Could not create optimization profile")
        print("❌ Dynamic batch size may not work properly")
        return False
    
    # Set the shape constraints
    try:
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        print(f"✅ Set shape constraints for {input_name}")
    except Exception as e:
        print(f"❌ Failed to set shape constraints: {e}")
        return False
    
    # Add profile to config
    try:
        config.add_optimization_profile(profile)
        print("✅ Added optimization profile to config")
    except Exception as e:
        print(f"❌ Failed to add optimization profile: {e}")
        return False
    
    print(f"Dynamic shapes configured:")
    print(f"  Min: {min_shape}")
    print(f"  Opt: {opt_shape}")
    print(f"  Max: {max_shape}")
    return True

def convert_to_trt(onnx_path, trt_path, precision_mode):
    """Converts an ONNX model to a TensorRT engine with dynamic batch support."""
    if os.path.exists(trt_path):
        print(f"✅ TensorRT engine already exists at: {trt_path}")
        return True

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    
    print(f"Building {precision_mode.upper()} engine for {onnx_path}...")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("❌ Failed to parse ONNX file. Errors:")
            for error in range(parser.num_errors):
                print(f"  - {parser.get_error(error)}")
            return False
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * (1024**3))  # 2GB workspace

    # CRITICAL: Set up dynamic shapes for batch size flexibility
    if not setup_dynamic_shapes(network, config, builder):
        print("⚠️ Warning: Dynamic shapes setup failed. Engine may have fixed batch size.")
        # Continue anyway, but warn user
    
    # Configure precision mode
    if precision_mode == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✅ FP16 precision enabled")
        else:
            print("⚠️ FP16 not supported on this platform.")
            return False
            
    elif precision_mode == 'int8':
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.FP16)  # Keep FP16 fallback
            
            calibrator = Int8Calibrator(CALIBRATION_IMAGE_DIR)
            config.int8_calibrator = calibrator
            print("✅ INT8 precision enabled with calibration")
        else:
            print("⚠️ INT8 not supported on this platform.")
            return False
    
    print("Building the TensorRT engine... (This may take several minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("❌ Failed to build the TensorRT engine.")
        return False
    
    with open(trt_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"✅ Engine built successfully and saved to: {trt_path}")
    print(f"✅ Engine supports dynamic batch sizes from {MIN_BATCH_SIZE} to {MAX_BATCH_SIZE}")
    return True

def verify_dynamic_engine(trt_path):
    """
    Verify that the built engine supports dynamic batch sizes.
    """
    if not os.path.exists(trt_path):
        return False
        
    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)
    
    with open(trt_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        print(f"❌ Failed to load engine from {trt_path}")
        return False
    
    # Check input binding
    input_idx = 0  # Assuming first binding is input
    if engine.get_tensor_shape(engine.get_tensor_name(input_idx))[0] == -1:
        print(f"✅ Engine supports dynamic batch sizes")
        return True
    else:
        print(f"❌ Engine has fixed batch size: {engine.get_tensor_shape(engine.get_tensor_name(input_idx))[0]}")
        return False

def main():
    """Main function to run the TensorRT conversion."""
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"❌ ONNX model not found: {ONNX_MODEL_PATH}")
        print("Please run the corrected ONNX export script first.")
        return

    print(f"Dynamic batch size configuration:")
    print(f"  Min batch size: {MIN_BATCH_SIZE}")
    print(f"  Optimal batch size: {OPT_BATCH_SIZE}")
    print(f"  Max batch size: {MAX_BATCH_SIZE}")

    # Convert to FP16
    print("=" * 60)
    if convert_to_trt(ONNX_MODEL_PATH, TRT_FP16_PATH, "fp16"):
        verify_dynamic_engine(TRT_FP16_PATH)
    
    # Convert to INT8
    print("=" * 60)
    if not os.path.exists(CALIBRATION_IMAGE_DIR) or not os.listdir(CALIBRATION_IMAGE_DIR):
        print(f"⚠️ WARNING: Calibration directory '{CALIBRATION_IMAGE_DIR}' is empty.")
        print("Skipping INT8 conversion. Please add images to the directory to enable it.")
    else:
        if convert_to_trt(ONNX_MODEL_PATH, TRT_INT8_PATH, "int8"):
            verify_dynamic_engine(TRT_INT8_PATH)

if __name__ == "__main__":
    main()