#!/bin/bash
#
# Full 4DGS Pipeline
# Processes multi-view video data and trains 4D Gaussian Splatting model
#

export CUDA_VISIBLE_DEVICES=1

set -e

# Configuration - Edit these paths
DATA_DIR="/home/phd/12/josmyfaure/dip_makeup/DIP"
OUTPUT_DIR="/home/phd/12/josmyfaure/dip_makeup/DIP/output"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create output directories
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "4DGS Pipeline - Starting"
echo "=============================================="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Activate conda environment (if needed)
# conda activate dip

# Step 1: Background Removal
echo "[Step 1/5] Background Removal..."
python "$SCRIPT_DIR/scripts/01_background_removal.py" \
    --input_dir "$DATA_DIR/DIP" \
    --bg_dir "$DATA_DIR/cali+bg/bg" \
    --output_dir "$OUTPUT_DIR/nobg"

# Step 2: Extract Frames
echo "[Step 2/5] Extracting Frames..."
python "$SCRIPT_DIR/scripts/02_extract_frames.py" \
    --input_dir "$OUTPUT_DIR/nobg" \
    --output_dir "$OUTPUT_DIR/frames" \
    --num_frames 10

# Step 3: Undistort Images
echo "[Step 3/5] Undistorting Images..."
python "$SCRIPT_DIR/scripts/03_undistort_images.py" \
    --input_dir "$OUTPUT_DIR/frames" \
    --calib_dir "$DATA_DIR/cali+bg/0" \
    --output_dir "$OUTPUT_DIR/undistorted" \
    --num_frames 10

# Step 4: Prepare Training Data
echo "[Step 4/5] Preparing Training Data..."
python "$SCRIPT_DIR/scripts/04_prepare_data.py" \
    --input_dir "$OUTPUT_DIR/undistorted" \
    --calib_dir "$DATA_DIR/cali+bg/0" \
    --output_dir "$OUTPUT_DIR/processed" \
    --num_frames 10

# Step 5: Train Model
echo "[Step 5/5] Training Model..."
python "$SCRIPT_DIR/train.py" \
    --data_dir "$OUTPUT_DIR/processed" \
    --output_dir "$OUTPUT_DIR/checkpoints" \
    --sequence \
    --start_frame 0 \
    --end_frame 10 \
    --group_size 2

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo "Checkpoints saved to: $OUTPUT_DIR/checkpoints"
echo ""
echo "To render results:"
echo "  python scripts/05_render_results.py \\"
echo "      --checkpoint $OUTPUT_DIR/checkpoints/0 \\"
echo "      --data_dir $OUTPUT_DIR/processed/0 \\"
echo "      --output_dir $OUTPUT_DIR/renders"
