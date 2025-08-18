import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import os
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer

# --- MODIFIED: Added TensorRT and PyCUDA imports ---
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

## 1. Configuration
MODEL_ID = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"
SAVED_MODEL_PATH = './finetuned_tinyclip_multilabel.pt'
VALIDATION_DIR = './valid_crops'
CLASS_LABELS = ["calyx", "fruitlet", "peduncle", "negative"]

# --- NEW: Define path to your TensorRT engine ---
TRT_ENGINE_PATH = 'tinyclip_int8_dynamic.trt'
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# --- BATCH SIZE FOR SINGLE INFERENCE ---
BATCH_SIZE = 1

## 2. Load Processors and Pre-compute Text Features
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# We still need the original model to get the text encoder and logit_scale
pt_model = CLIPModel.from_pretrained(MODEL_ID)
pt_model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=device))
pt_model.to(device)
pt_model.eval()

processor = CLIPProcessor.from_pretrained(MODEL_ID)
text_prompts = [f"a photo of a {label}" for label in CLASS_LABELS]
print(f"Testing against prompts: {text_prompts}")

# --- NEW: Pre-compute text features using the PyTorch model ---
print("Pre-computing text features...")
with torch.no_grad():
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
    text_features = pt_model.get_text_features(**text_inputs)
    # Normalize text features
    text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
    # Get the learned logit scale
    logit_scale = pt_model.logit_scale.exp()
print("Text features computed.")

# --- NEW: Helper class for managing TensorRT memory ---
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# --- FIXED: Load TensorRT Engine and Allocate Buffers ---
print(f"Loading TensorRT engine from: {TRT_ENGINE_PATH}")
if not os.path.exists(TRT_ENGINE_PATH):
    raise FileNotFoundError(f"TensorRT engine not found at '{TRT_ENGINE_PATH}'")

runtime = trt.Runtime(TRT_LOGGER)
with open(TRT_ENGINE_PATH, 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# --- FIXED: Proper memory allocation for dynamic batch engines ---
inputs, outputs, bindings, stream = [], [], [], cuda.Stream()

print("Setting up memory buffers...")
for i in range(engine.num_io_tensors):
    tensor_name = engine.get_tensor_name(i)
    tensor_mode = engine.get_tensor_mode(tensor_name)
    tensor_dtype = engine.get_tensor_dtype(tensor_name)
    tensor_shape = engine.get_tensor_shape(tensor_name)
    
    print(f"Tensor {i}: {tensor_name}, Mode: {tensor_mode}, Shape: {tensor_shape}, Dtype: {tensor_dtype}")
    
    # For dynamic batch engines, set the actual batch size we'll use
    if tensor_mode == trt.TensorIOMode.INPUT:
        # Set input shape for single inference (batch_size=1)
        actual_shape = (BATCH_SIZE, 3, 224, 224)  # Assuming standard CLIP input
        context.set_input_shape(tensor_name, actual_shape)
        print(f"Set input shape for {tensor_name}: {actual_shape}")
    
    # Get the actual shape after setting input shapes
    if tensor_mode == trt.TensorIOMode.INPUT:
        actual_shape = context.get_tensor_shape(tensor_name)
    else:
        actual_shape = context.get_tensor_shape(tensor_name)
    
    # Calculate memory size based on actual shape
    size = trt.volume(actual_shape)
    dtype = trt.nptype(tensor_dtype)
    
    print(f"Allocating {size} elements of type {dtype} for {tensor_name}")
    
    # Allocate host and device memory
    try:
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        
        if tensor_mode == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
            print(f"✅ Input buffer allocated: {host_mem.nbytes} bytes")
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            print(f"✅ Output buffer allocated: {host_mem.nbytes} bytes")
            
    except Exception as e:
        print(f"❌ Memory allocation failed for {tensor_name}: {e}")
        raise

print("TensorRT engine loaded and buffers allocated.")

## 3. Evaluation
all_true_labels = []
all_predicted_labels = []
all_pred_scores = []

if not os.path.isdir(VALIDATION_DIR):
    print(f"Error: Validation directory '{VALIDATION_DIR}' not found.")
else:
    pbar = tqdm(os.walk(VALIDATION_DIR), desc="Evaluating with TensorRT")
    for root, _, files in pbar:
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(root, file)
            true_label = os.path.basename(root)

            image = Image.open(image_path).convert("RGB")
            
            # --- MODIFIED: Use processor only for the image ---
            image_input = processor(images=image, return_tensors="pt")
            pixel_values = image_input['pixel_values'].numpy()

            # --- FIXED: TensorRT Inference execution ---
            # Copy input data to the GPU
            np.copyto(inputs[0].host[:pixel_values.size], pixel_values.ravel())
            cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, stream)
            
            # Run inference
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            
            # Copy output back to the host
            cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, stream)
            stream.synchronize()
            
            # Get the image features from the TRT engine output
            output_shape = context.get_tensor_shape(engine.get_tensor_name(engine.num_io_tensors - 1))
            feature_size = output_shape[-1]  # Last dimension should be feature size (512)
            
            raw_output = torch.from_numpy(outputs[0].host[:feature_size]).to(device)
            image_features = raw_output.reshape(1, feature_size)
            
            # --- MODIFIED: Manually calculate logits ---
            # Normalize image features
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            # Calculate cosine similarity
            logits_per_image = logit_scale * image_features @ text_features.t()
            
            predicted_index = logits_per_image.argmax().item()
            predicted_label = CLASS_LABELS[predicted_index]

            all_true_labels.append(true_label)
            all_predicted_labels.append(predicted_label)
            probs = logits_per_image.softmax(dim=1).cpu().numpy()
            all_pred_scores.append(probs[0])

            pbar.set_postfix({"Correct": f"{len([i for i, j in zip(all_predicted_labels, all_true_labels) if i == j])}/{len(all_true_labels)}"})

# The rest of the script (reporting and plotting) remains exactly the same.
if all_true_labels:
    print("\n--- Evaluation Complete ---")
    report = classification_report(all_true_labels, all_predicted_labels, labels=CLASS_LABELS, digits=4)
    print("--- Classification Report ---")
    print(report)

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(all_true_labels, all_predicted_labels, labels=CLASS_LABELS)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                  xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # --- Added Normalized Confusion Matrix ---
    print("\n--- Normalized Confusion Matrix ---")
    cm_normalized = confusion_matrix(all_true_labels, all_predicted_labels, labels=CLASS_LABELS, normalize='true')

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                  xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    plt.show()

    # Binarize labels and convert scores for plotting
    lb = LabelBinarizer()
    lb.fit(CLASS_LABELS)
    y_true_bin = lb.transform(all_true_labels)
    y_scores = np.array(all_pred_scores)

    # --- FIGURE 1: All P-R curves on one plot with an inset ---
    print("\n--- Combined Precision-Recall Curves ---")
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, class_name in enumerate(CLASS_LABELS):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        # --- MODIFIED: Linewidth is thicker only for 'calyx' ---
        linewidth = 4 if class_name == 'calyx' else 2
        ax.plot(recall, precision, lw=linewidth, label=f'P-R curve for {class_name}')

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves for the TinyCLIP Classifier")
    ax.legend(loc="best")
    ax.grid(alpha=0.4)
    ax.set_xlim(0.75, 1.05)
    ax.set_ylim(0.75, 1.05)

    # --- ADD INSET PLOT FOR CONTEXT ---
    ax_inset = fig.add_axes([0.18, 0.18, 0.4, 0.4])
    for i, class_name in enumerate(CLASS_LABELS):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        # --- MODIFIED: Inset linewidth is also thicker only for 'calyx' ---
        linewidth = 4 if class_name == 'calyx' else 2
        ax_inset.plot(recall, precision, lw=linewidth)
    ax_inset.set_title("Full Range (0-1)")
    ax_inset.set_xlabel("Recall")
    ax_inset.set_ylabel("Precision")
    ax_inset.grid(alpha=0.4)

    plt.show()

    # --- FIGURE 2: P-R curve for each class as a separate subplot with insets ---
    print("\n--- Per-Class Precision-Recall Curves ---")
    fig, axes = plt.subplots(1, len(CLASS_LABELS), figsize=(20, 6), sharey=True)
    fig.suptitle('Per-Class Precision-Recall Curves', fontsize=16)

    for i, class_name in enumerate(CLASS_LABELS):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        # --- REVERTED: All lines have the same standard thickness ---
        axes[i].plot(recall, precision, lw=2, label=f'P-R for {class_name}')
        axes[i].set_title(f'Class: {class_name}')
        axes[i].set_xlabel('Recall')
        axes[i].set_ylabel('Precision')
        axes[i].grid(alpha=0.4)
        axes[i].legend()
        axes[i].set_xlim(0.75, 1.05)
        axes[i].set_ylim(0.75, 1.05)

        # --- ADD INSET PLOT FOR EACH SUBPLOT ---
        ax_inset = axes[i].inset_axes([0.1, 0.1, 0.5, 0.5])
        ax_inset.plot(recall, precision, lw=2)
        ax_inset.set_title("Full Range")
        ax_inset.set_xlabel("R", fontsize=8)
        ax_inset.set_ylabel("P", fontsize=8)
        ax_inset.grid(alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

else:
    print("No images found in the validation directory.")