# 4D Gaussian Splatting for Multi-View Video Reconstruction

This project implements a complete pipeline for reconstructing dynamic 3D scenes from multi-view video using 4D Gaussian Splatting (4DGS) with the V³ framework.

## Project Structure

```
4DGS_project/
├── README.md
├── requirements.txt
├── scripts/
│   ├── 01_background_removal.py    # Remove backgrounds using BackgroundMattingV2
│   ├── 02_extract_frames.py        # Extract frames from videos
│   ├── 03_undistort_images.py      # Undistort images using camera calibration
│   ├── 04_prepare_data.py          # Format data for V3 training
│   └── 05_render_results.py        # Render novel views from trained model
├── train.py                        # Training script
├── config.py                       # Configuration settings
└── run_pipeline.sh                 # Full pipeline script
```

## Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA
- V³ Framework (4DGS)
- BackgroundMattingV2

## Installation

```bash
# Clone V³ framework
git clone https://github.com/AuthorityWang/VideoGS
cd VideoGS
pip install -r requirements.txt

# Install submodules
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

# Install other dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
# Set paths in config.py, then run full pipeline:
./run_pipeline.sh
```

### Step-by-Step

#### 1. Background Removal
```bash
python scripts/01_background_removal.py \
    --input_dir data/videos \
    --bg_dir data/backgrounds \
    --output_dir output/nobg
```

#### 2. Extract Frames
```bash
python scripts/02_extract_frames.py \
    --input_dir output/nobg \
    --output_dir output/frames \
    --num_frames 300
```

#### 3. Undistort Images
```bash
python scripts/03_undistort_images.py \
    --input_dir output/frames \
    --calib_dir data/calibration \
    --output_dir output/undistorted
```

#### 4. Prepare Training Data
```bash
python scripts/04_prepare_data.py \
    --input_dir output/undistorted \
    --calib_dir data/calibration \
    --output_dir output/processed
```

#### 5. Train Model
```bash
python train.py \
    --data_dir output/processed \
    --output_dir output/checkpoints \
    --iterations 16000
```

#### 6. Render Results
```bash
python scripts/05_render_results.py \
    --checkpoint output/checkpoints/0 \
    --output_dir output/renders
```

## Data Format

### Input
- `data/videos/`: MP4 videos from each camera (e.g., `001001.mp4`, `002001.mp4`, ...)
- `data/backgrounds/`: Background images for each camera
- `data/calibration/`: COLMAP calibration files (`cameras.bin`, `images.bin`, `points3D.bin`)

### Output
- `output/checkpoints/`: Trained model checkpoints (PLY files)
- `output/renders/`: Rendered images and comparison figures

## Configuration

Edit `config.py` to set paths and parameters:

```python
# Paths
V3_PATH = "/path/to/V3"
DATA_DIR = "/path/to/data"
OUTPUT_DIR = "/path/to/output"

# Training parameters
ITERATIONS = 16000
GROUP_SIZE = 20
WHITE_BACKGROUND = True
```

## Acknowledgments

- [V³ Framework](https://github.com/pxxxl/V3) - SIGGRAPH Asia 2024
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2)
