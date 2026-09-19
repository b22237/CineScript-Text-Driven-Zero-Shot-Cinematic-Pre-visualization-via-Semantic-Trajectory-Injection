# Garment Motion Generation Pipeline
- [Report](https://drive.google.com/file/d/1fzX62m4dyMt0K62_9bFGCrr2Z8zokoIQ/view?usp=sharing)

A multi-component pipeline for image segmentation, editing, and motion generation combining **Segment Anything (SAM)**, **InstructPix2Pix**, and **MotionModes**.

## 🎯 Overview

This project integrates three powerful AI models into a unified Gradio-based application:

| Component | Description |
|-----------|-------------|
| **Segment Anything (SAM)** | High-quality object segmentation from Meta AI |
| **InstructPix2Pix** | Instruction-based image editing |
| **MotionModes** | Motion generation and discovery (CVPR 2025) |

### Pipeline Flow

```
Input Image → [Optional: InstructPix2Pix Edit] → SAM Segmentation → Motion Generation → Output Video
```

The main application (`motion_discovery_app.py`) orchestrates this entire pipeline through a Gradio web interface.

## 📁 Project Structure

```
├── segment-anything/     # SAM segmentation module
├── instruct-pix2pix/     # Image editing with text instructions
├── MotionModes/          # Motion generation pipeline (contains main app)
│   └── motion_discovery_app.py  # 🎯 MAIN APPLICATION
└── environment.yaml      # Unified conda environment
```

## 🚀 Installation

### Prerequisites

- CUDA-capable GPU (recommended: VRAM > 16GB)
- Conda package manager
- Git LFS (for model downloads)

### Step 1: Create Conda Environment

```bash
conda env create -f environment.yaml
conda activate motion-pipeline
```

### Step 2: Install Segment Anything

```bash
cd segment-anything
pip install -e .
pip install opencv-python pycocotools matplotlib
```

### Step 3: Download Model Checkpoints

#### SAM Checkpoints

Download SAM model checkpoints from the official repository:

```bash
# ViT-H (Huge) - Best quality
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# ViT-B (Base) - Faster inference
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# ViT-L (Large) - Balanced
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

#### InstructPix2Pix Checkpoint

```bash
cd instruct-pix2pix
bash scripts/download_checkpoints.sh
```

#### MotionModes / Motion-I2V Models

```bash
git clone https://huggingface.co/wangfuyun/Motion-I2V
```

## 📖 Usage

### 🎯 Main Application (Recommended)

The `motion_discovery_app.py` is the **main entry point** that runs the complete pipeline with a Gradio web interface:

```bash
cd MotionModes
conda activate motion-pipeline
python motion_discovery_app.py --share
```

**Options:**
- `--server-name`: Server address (default: `127.0.0.1`)
- `--server-port`: Port number (default: `7860`)
- `--share`: Create a public Gradio link

The app provides:
1. **Image Upload** - Upload your source image
2. **Optional Editing** - Edit image with text instructions (InstructPix2Pix)
3. **Automatic Segmentation** - SAM generates masks automatically
4. **Motion Generation** - Generate motion videos with text prompts

#### Motion Prompt Keywords

The TrajectoryPlanner supports these motion directions:
- `left`, `right`, `up`, `down`
- `zoom_in`, `zoom_out`

Example prompts:
- *"move the garment to the left"*
- *"zoom in on the dress"*
- *"flowing movement to the right"*

---

### Individual Components

#### 1. Segment Anything (SAM)

Run segmentation on an image:

```bash
cd segment-anything
python run.py <path_to_image>
```

Example:
```bash
python run.py ./imgs/example.jpg
```

Output masks are saved to `segment-anything/masks/mask_000.png`.

#### 2. InstructPix2Pix

##### Edit a single image via CLI:

```bash
cd instruct-pix2pix
conda activate ip2p
python edit_cli.py --input imgs/example.jpg --output imgs/output.jpg --edit "turn it into a red dress"
```

##### Parameters for fine-tuning:

```bash
python edit_cli.py \
    --steps 100 \
    --resolution 512 \
    --seed 1371 \
    --cfg-text 7.5 \
    --cfg-image 1.2 \
    --input imgs/example.jpg \
    --output imgs/output.jpg \
    --edit "make the garment blue"
```

##### Launch interactive Gradio app:

```bash
python edit_app.py
```

#### 3. MotionModes (Standalone)

```bash
cd MotionModes
conda activate garment-pipeline
```

1. Set input data path in `frame_data.json` (see example setup in the file)

2. Run motion discovery (standalone, without the full pipeline):
```bash
python motion_discovery.py
```

## ⚙️ Configuration

### SAM Model Types

| Model | Checkpoint | VRAM | Speed |
|-------|------------|------|-------|
| `vit_h` | `sam_vit_h_4b8939.pth` | ~16GB | Slowest |
| `vit_l` | `sam_vit_l_0b3195.pth` | ~12GB | Medium |
| `vit_b` | `sam_vit_b_01ec64.pth` | ~8GB | Fastest |

### InstructPix2Pix Tips

- **cfg-text**: Higher values = follow instruction more closely (default: 7.5)
- **cfg-image**: Higher values = preserve original image more (default: 1.5)
- **steps**: More steps = better quality but slower (default: 100)

## 📦 Dependencies

Key dependencies across all modules:

- Python 3.8-3.10
- PyTorch >= 1.11.0
- CUDA 11.3+ / 11.7+
- transformers
- diffusers
- gradio
- opencv-python
- einops

## 🔗 References

### Segment Anything
```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

### InstructPix2Pix
```bibtex
@article{brooks2022instructpix2pix,
  title={InstructPix2Pix: Learning to Follow Image Editing Instructions},
  author={Brooks, Tim and Holynski, Aleksander and Efros, Alexei A},
  journal={arXiv preprint arXiv:2211.09800},
  year={2022}
}
```

### MotionModes
```bibtex
@article{pandey2025motionmodes,
  title={Motion Modes: What Could Happen Next?},
  author={Pandey, Karran and Gadelha, Matheus and Hold-Geoffroy, Yannick and Singh, Karan and Mitra, Niloy J. and Guerrero, Paul},
  journal={CVPR 2025},
  year={2025}
}
```

### Motion-I2V
```bibtex
@article{shi2024motion,
  title={Motion-i2v: Consistent and controllable image-to-video generation with explicit motion modeling},
  author={Shi, Xiaoyu and Huang, Zhaoyang and Wang, Fu-Yun and Bian, Weikang and Li, Dasong and Zhang, Yi and Zhang, Manyuan and Cheung, Ka Chun and See, Simon and Qin, Hongwei and others},
  journal={SIGGRAPH 2024},
  year={2024}
}
```

## 📄 License

This project combines components under various licenses:
- **Segment Anything**: Apache 2.0 License
- **InstructPix2Pix**: MIT License  
- **MotionModes**: See LICENSE.md

## 🤝 Acknowledgements

- [Meta AI Research](https://ai.facebook.com/research/) for Segment Anything
- [Tim Brooks et al.](https://www.timothybrooks.com/instruct-pix2pix/) for InstructPix2Pix
- [Adobe Research](https://research.adobe.com/) for MotionModes
