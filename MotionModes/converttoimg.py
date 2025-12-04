import os
from PIL import Image, ImageSequence

def gif_to_images(gif_path, output_folder):
    # 1. Check if the GIF file exists
    if not os.path.exists(gif_path):
        print(f"Error: The file '{gif_path}' was not found.")
        return

    # 2. Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        # 3. Open the GIF
        with Image.open(gif_path) as im:
            print(f"Processing '{gif_path}'...")

            # 4. Iterate over frames
            # ImageSequence.Iterator allows us to loop through frames easily
            for index, frame in enumerate(ImageSequence.Iterator(im)):
                
                # Construct the output filename (e.g., frame_0.png, frame_1.png)
                output_filename = f"frame_{index}.png"
                output_path = os.path.join(output_folder, output_filename)
                
                # 5. Convert to RGBA (keeps transparency) or RGB
                # This ensures colors render correctly even if the GIF is in Palette mode
                frame = frame.convert("RGBA")
                
                # 6. Save the frame
                frame.save(output_path)
                print(f"Saved: {output_filename}")
                
            print(f"\n✅ Done! Extracted {index + 1} frames to '{output_folder}'")

    except Exception as e:
        print(f"An error occurred: {e}")

# --- Usage ---
# Replace 'animation.gif' with the path to your GIF file
gif_file = "/home/soham/garments/preet/new/MotionModes/outputs/motion_discovery_gradio/20251126_031512/generation_1/gradio/samples/test_out_2.gif" 
output_dir = "/home/soham/garments/preet/new/MotionModes/giftoimg"

gif_to_images(gif_file, output_dir)