import os
import sys
import torch
import numpy as np
import gradio as gr
from PIL import Image
from datetime import datetime
import json
import argparse
import subprocess
import gc

# --- ENVIRONMENT CONFIGURATION ---
# Fixes "Unrecognized CachingAllocator" on older PyTorch versions
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Import from motion_discovery.py
from motion_discovery import (
    Drag,
    visualize_drag_v2,
)

# Global settings
drag_model = None
output_base_dir = "outputs/motion_discovery_gradio"
os.makedirs(output_base_dir, exist_ok=True)

# ==========================================
# 1. MEMORY MANAGER
# ==========================================
def cleanup_gpu():
    """Unload Motion Model to free VRAM for subprocesses."""
    global drag_model
    
    if drag_model is not None:
        print("🧹 [GPU SWAP] Unloading Motion Model...")
        del drag_model
        drag_model = None
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Safe IPC collect (ignoring errors if not supported)
        try:
            torch.cuda.ipc_collect()
        except:
            pass
    print("✨ GPU is clean.")
class TrajectoryPlanner:
    """
    NOVELTY MODULE: Converts text prompts into explicit motion vectors.
    """
    def __init__(self):
        self.directions = {
            "left":  [-1, 0], "right": [1, 0],
            "up":    [0, -1], "down":  [0, 1],
            "zoom_in": "spread", "zoom_out": "contract"
        }

    def generate_points(self, mask_pil, text_prompt, num_points=5):
        # 1. Parse Text
        prompt = text_prompt.lower()
        motion_type = "random"
        vector = [0, 0]
        
        for key, vec in self.directions.items():
            if key in prompt:
                motion_type = key
                vector = vec
                break
        
        if motion_type == "random":
            print(f"⚠️ TrajectoryPlanner: No direction found in '{text_prompt}'.")
            return []

        # 2. Analyze Mask
        mask_np = np.array(mask_pil)
        # Get coordinates of the object (white pixels > 127)
        y_indices, x_indices = np.where(mask_np > 127)
        
        if len(y_indices) == 0: return []

        tracking_points = []
        scale = 20 # How far the pixels move
        
        # 3. Generate Vectors
        for _ in range(num_points):
            idx = np.random.randint(0, len(y_indices))
            start_y, start_x = y_indices[idx], x_indices[idx]
            
            if motion_type in ["zoom_in", "zoom_out"]:
                # Radial logic relative to center
                h, w = mask_np.shape
                center_y, center_x = h//2, w//2
                vec_y, vec_x = start_y - center_y, start_x - center_x
                mag = np.sqrt(vec_y**2 + vec_x**2) + 1e-5
                direction = 1 if motion_type == "zoom_in" else -1
                
                end_y = int(start_y + (vec_y/mag) * scale * direction)
                end_x = int(start_x + (vec_x/mag) * scale * direction)
            else:
                # Linear logic
                end_y = int(start_y + vector[1] * scale)
                end_x = int(start_x + vector[0] * scale)
                
            tracking_points.append([start_x, start_y, end_x, end_y])
            
        print(f"✅ TrajectoryPlanner: Injected {len(tracking_points)} points for '{motion_type}'")
        return tracking_points

# Initialize Planner Global
planner = TrajectoryPlanner()
def initialize_model():
    """Loads the heavy Motion Model only when needed."""
    global drag_model
    if drag_model is None:
        print("🚀 [GPU SWAP] Loading Motion Discovery Model...")
        drag_model = Drag(
            device="cuda:0",
            pretrained_model_path="/DATA/soham/preet/data/Motionmodes/Motion-I2V/models/stage1/StableDiffusion-FlowGen",
            inference_config="configs/configs_flowgen/inference/inference.yaml",
            height=320,
            width=512,
            model_length=16,
        )
        print("✅ Model loaded!")
    return drag_model

# ==========================================
# 2. SUBPROCESS CALLS
# ==========================================
def run_segment_anything(image_path):
    """Run the segment-anything script (run.py) via subprocess."""
    print(f"🚀 Starting SAM Subprocess on {image_path}...")
    segment_script_path = "/home/soham/garments/preet/new/segment-anything/run.py"
    
    # GPU should be free here due to cleanup_gpu()
    cmd = ["python", segment_script_path, image_path]
    
    try:
        subprocess.run(cmd, check=True)
        mask_path = os.path.join("/home/soham/garments/preet/new/segment-anything/masks", "mask_000.png")
        if os.path.exists(mask_path):
            print("✅ SAM Mask generated successfully!")
            return Image.open(mask_path).convert("L")
        else:
            print("❌ Mask not found after generation.")
            return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running SAM: {e}")
        return None

def run_pix2pix_edit(image_pil, edit_instruction):
    """Runs InstructPix2Pix subprocess."""
    print(f"\n🎨 [DEBUG] Starting InstructPix2Pix...")
    
    PARALLEL_FOLDER_NAME = "instruct-pix2pix" 
    ENV_PYTHON_PATH = "/home/soham/.conda/envs/project/bin/python"

    temp_dir = "temp_edit"
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    
    input_filename = f"input_{timestamp}.jpg"
    output_filename = f"output_{timestamp}.jpg"
    abs_input_path = os.path.abspath(os.path.join(temp_dir, input_filename))
    abs_output_path = os.path.abspath(os.path.join(temp_dir, output_filename))
    
    try:
        image_pil.save(abs_input_path)
    except Exception as e:
        print(f"❌ Save failed: {e}")
        return image_pil

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    pix2pix_dir = os.path.join(parent_dir, PARALLEL_FOLDER_NAME)
    
    my_env = os.environ.copy()
    if "PYTHONPATH" in my_env: del my_env["PYTHONPATH"]

    cmd = [ENV_PYTHON_PATH, "edit_cli.py", "--input", abs_input_path, "--output", abs_output_path, "--edit", edit_instruction]
    
    try:
        subprocess.run(cmd, cwd=pix2pix_dir, env=my_env, capture_output=True, text=True, check=True)
        if os.path.exists(abs_output_path):
            edited_image = Image.open(abs_output_path).convert("RGB")
            edited_image.load() 
            return edited_image
    except Exception as e:
        print(f"❌ Edit failed: {e}")
        return image_pil
    return image_pil

def preprocess_image(image_pil, target_width=512, target_height=320):
    raw_w, raw_h = image_pil.size
    resize_ratio = max(target_width / raw_w, target_height / raw_h)
    image_pil = image_pil.resize((int(raw_w * resize_ratio), int(raw_h * resize_ratio)), Image.BILINEAR)
    from torchvision import transforms
    image_pil = transforms.CenterCrop((target_height, target_width))(image_pil)
    return image_pil

# ==========================================
# 3. GENERATION LOOP (With RESTORED PREVIEWS)
# ==========================================
def generate_motion_video(
    input_image, prompt, enable_editing, edit_instruction,
    num_generations=1, num_samples=3, num_inference_steps=25,
    guidance_scale=7.0, use_physics_guidance=True,
    dilate_mask=True, dilation_iterations=1, save_intermediate=False,
    progress=gr.Progress()
):
    
    if input_image is None:
        yield None, [], "❌ **Error:** Input image required!", None, None, None
        return

    final_display_image = input_image
    model = None 

    try:
        # STEP 1: EMPTY THE GPU
        progress(0.05, desc="🧹 Unloading models for heavy tasks...")
        cleanup_gpu()

        # STEP 2: EDITING
        if enable_editing:
            progress(0.1, desc="🎨 Running InstructPix2Pix...")
            edited_image = run_pix2pix_edit(final_display_image, edit_instruction)
            if edited_image:
                final_display_image = edited_image
                # YIELD: Update "Processed Input" Preview immediately
                yield (None, None, "✅ **Edit Complete.** Starting Masking...", None, final_display_image, None)

        # STEP 3: MASKING
        progress(0.2, desc="✂️ Running Segmentation...")
        temp_image_path = "/tmp/temp_sam_input.jpg"
        final_display_image.save(temp_image_path)
        
        mask_image = run_segment_anything(temp_image_path)
        
        # YIELD: Update "Mask" Preview immediately
        yield (None, None, "✅ **Mask Ready.** Loading Motion Model...", None, final_display_image, mask_image)
        
        # STEP 4: MOTION GENERATION
        progress(0.3, desc="🚀 Reloading Motion Model...")
        model = initialize_model()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_base_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # --- CRITICAL FIX: Resize everything to model specs first ---
        image_pil = preprocess_image(final_display_image) 
        
        if mask_image:
             mask_pil = mask_image.resize((512, 320)) # Resize mask to match image
        else:
             mask_pil = Image.new("L", (512, 320), 0)

        # Save assets
        image_pil.save(os.path.join(output_dir, "input_frame.png"))
        mask_pil.save(os.path.join(output_dir, "mask.png"))
        temp_frame_path = os.path.join(output_dir, "first_frame.png")
        image_pil.save(temp_frame_path)
        
        # Prepare 3D mask for model
        mask_3d = np.stack([np.array(mask_pil)]*3, axis=2)
        
        # --- NOVELTY INJECTION ---
        # We pass the RESIZED mask (512x320) so coordinates match the video size
        progress(0.4, desc="🧠 Calculating Trajectories...")
        auto_points = planner.generate_points(mask_pil, prompt)
        
        all_video_paths = []
        
        progress(0.5, desc="🎬 Generating Videos...")
        for gen_idx in range(num_generations):
            gen_output_dir = os.path.join(output_dir, f"generation_{gen_idx + 1}")
            os.makedirs(gen_output_dir, exist_ok=True)
            
            _, output_path = model.run(
                first_frame_path=temp_frame_path,
                brush_mask=mask_3d,
                tracking_points=auto_points,  # <--- FIXED: USE THE POINTS HERE!
                inference_batch_size=1,
                flow_unit_id=4,
                prompt=prompt,
                output_dir=gen_output_dir,
            )
            
            # Locate output (Handle MotionModes weird output paths)
            if output_path and os.path.exists(output_path):
                all_video_paths.append(output_path)
            
            # Check for samples folder if main path is empty
            samples_dir = os.path.join(gen_output_dir, "gradio", "samples")
            if os.path.exists(samples_dir):
                for f in os.listdir(samples_dir):
                    if f.endswith(".mp4") or f.endswith(".gif"):
                        all_video_paths.append(os.path.join(samples_dir, f))

        # STEP 5: CLEANUP
        del model
        model = None
        progress(0.95, desc="🧹 Final Cleanup...")
        cleanup_gpu()

        info_text = f"✅ **Complete!** Saved to `{output_dir}`"
        if len(auto_points) > 0:
            info_text += f"\n🎯 **Auto-Director:** Injected {len(auto_points)} trajectory points based on prompt."
        
        # YIELD: Final Results
        yield (
            all_video_paths[0] if all_video_paths else None, 
            all_video_paths, 
            info_text, 
            output_dir, 
            final_display_image, 
            mask_image
        )
        
    except Exception as e:
        import traceback
        if 'model' in locals() and model is not None: del model
        cleanup_gpu()
        error_msg = f"❌ **Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        yield None, [], error_msg, None, final_display_image, None
# Gradio Interface
def create_gradio_interface():
    with gr.Blocks(title="Motion Discovery (Restored Features)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎬 Motion Discovery")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Source Image", type="pil", height=320)
                enable_editing = gr.Checkbox(label="Enable Image Editing", value=True)
                edit_instruction = gr.Textbox(label="Edit Instruction", value="turn him into a cyborg")
                prompt = gr.Textbox(label="Motion Prompt", lines=3)
                generate_btn = gr.Button("🎬 Generate", variant="primary", size="lg")
                
                num_generations = gr.Slider(1, 10, value=1, visible=False) 
                
            with gr.Column(scale=1):
                # RESTORED: Edited Preview
                edited_preview = gr.Image(label="Processed Input (After Edit)", interactive=False, height=320)
                # RESTORED: Mask Preview
                mask_preview = gr.Image(label="Generated Mask", interactive=False, height=200)
                
                video_preview = gr.Video(label="Final Video", height=400)
                info_output = gr.Markdown()
                output_dir_text = gr.Textbox(visible=False)
                all_videos = gr.File(visible=False)

        generate_btn.click(
            fn=generate_motion_video,
            inputs=[input_image, prompt, enable_editing, edit_instruction, num_generations],
            outputs=[video_preview, all_videos, info_output, output_dir_text, edited_preview, mask_preview]
        )
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    
    print(f"\n⚡ App Starting (Features Restored)...\n")
    demo = create_gradio_interface()
    demo.queue(max_size=2).launch(server_name=args.server_name, server_port=args.server_port, share=True)