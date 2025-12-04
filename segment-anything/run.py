import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Segment an image using SAM Predictor.')
parser.add_argument('image_path', type=str, help='Path to the input image file')
args = parser.parse_args()

# --- SETUP: GPU & Model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Switch to 'vit_b' (Base) for speed
checkpoint_path = "/DATA/soham/preet/data/sam2/sam_vit_b_01ec64.pth" 

if not os.path.exists(checkpoint_path):
    print(f"Warning: Checkpoint not found at {checkpoint_path}. Checking fallback...")
    model_type = "vit_h"
    checkpoint_path = "/DATA/soham/preet/data/sam2/sam_vit_h_4b8939.pth"
else:
    model_type = "vit_b"

print(f"Loading model: {model_type}...")
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device=device)

# --- 1. INITIALIZE PREDICTOR ---
predictor = SamPredictor(sam)

# Verify image path
img_path = args.image_path
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")

img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not read image: {img_path}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- 2. SET IMAGE ---
print("Setting image...")
predictor.set_image(img)

# --- 3. CREATE PROMPTS (Center + Nearby Points) ---
h, w = img.shape[:2]
cx, cy = w // 2, h // 2

# Offset: 15% of the height
offset_y = int(h * 0.05) 

input_point = np.array([
    [cx, cy],              # 1. The exact Center
       # 2. Slightly Above (Upper body/Top)
    [cx+ offset_y, cy + offset_y]] )   # 3. Slightly Below (Lower body/Bottom)
   

# Label '1' means foreground (include this)
input_label = np.array([1,  1])

print(f"Predicting mask using 3 points (Center, Top-Mid, Bot-Mid)...")

# --- 4. PREDICT ---
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True, 
)

# --- 5. SELECT BEST MASK ---
# We generally prefer the mask with the highest IoU score
best_idx = np.argmax(scores)
final_mask = masks[best_idx].astype(np.uint8)

print(f"Selected mask index {best_idx} with score {scores[best_idx]:.3f}")

# --- SAVE OUTPUT ---
mask_dir = "/home/soham/garments/preet/new/segment-anything/masks"
os.makedirs(mask_dir, exist_ok=True)

# Always save as mask_000.png so the main app can find it easily
output_path = os.path.join(mask_dir, "mask_000.png")

# Save mask (0=Black, 255=White)
mask_uint8 = final_mask * 255
cv2.imwrite(output_path, mask_uint8)

print(f"✅ Mask saved to: {output_path}")