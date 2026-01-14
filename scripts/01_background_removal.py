#!/usr/bin/env python3
"""
Step 1: Background Removal

Removes background from multi-view videos using BackgroundMattingV2.
Outputs videos with white background for 4DGS training.

Usage:
    python scripts/01_background_removal.py \
        --input_dir data/videos \
        --bg_dir data/backgrounds \
        --output_dir output/nobg
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BGMATTING_PATH, CAMERA_IDS

# Add BackgroundMattingV2 to path
sys.path.insert(0, BGMATTING_PATH)


def load_model(model_type="resnet50", device="cuda"):
    """Load BackgroundMattingV2 model."""
    from model import MattingRefine
    
    model = MattingRefine(
        backbone=model_type,
        backbone_scale=1/4,
        refine_mode='sampling',
        refine_sample_pixels=80000
    )
    
    # Load pretrained weights
    weights_path = os.path.join(BGMATTING_PATH, f"pytorch_{model_type}.pth")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"Warning: Weights not found at {weights_path}")
    
    model = model.to(device).eval()
    return model


def process_video(model, video_path, bg_path, output_path, device="cuda"):
    """Process a single video with background removal."""
    import cv2
    
    # Load background image
    bgr = cv2.imread(bg_path)
    if bgr is None:
        print(f"Error: Cannot load background {bg_path}")
        return False
    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bgr_tensor = torch.from_numpy(bgr).permute(2, 0, 1).float() / 255.0
    bgr_tensor = bgr_tensor.unsqueeze(0).to(device)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # White background for compositing
    white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    with torch.no_grad():
        for _ in tqdm(range(total_frames), desc=os.path.basename(video_path)):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to tensor
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            src_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            src_tensor = src_tensor.unsqueeze(0).to(device)
            
            # Run matting
            pha, fgr = model(src_tensor, bgr_tensor)[:2]
            
            # Composite on white background
            pha_np = pha[0].permute(1, 2, 0).cpu().numpy()
            fgr_np = fgr[0].permute(1, 2, 0).cpu().numpy()
            
            composite = fgr_np * pha_np + (1 - pha_np) * (white_bg / 255.0)
            composite = (composite * 255).clip(0, 255).astype(np.uint8)
            
            # Write frame
            out.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    
    cap.release()
    out.release()
    return True


def main():
    parser = argparse.ArgumentParser(description="Background removal for multi-view videos")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input videos")
    parser.add_argument("--bg_dir", type=str, required=True, help="Directory with background images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "resnet101", "mobilenetv2"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading BackgroundMattingV2 ({args.model})...")
    model = load_model(args.model, args.device)
    
    # Process each camera
    for cam_id in CAMERA_IDS:
        video_name = f"{cam_id}001.mp4"
        video_path = os.path.join(args.input_dir, video_name)
        # Background files are named like 001001.png (same as video without extension)
        bg_path = os.path.join(args.bg_dir, f"{cam_id}001.png")
        output_path = os.path.join(args.output_dir, video_name)
        
        if not os.path.exists(video_path):
            print(f"Skipping {video_name} (not found)")
            continue
        
        if not os.path.exists(bg_path):
            # Try alternative naming: just cam_id
            bg_path = os.path.join(args.bg_dir, f"{cam_id}.png")
            if not os.path.exists(bg_path):
                bg_path = os.path.join(args.bg_dir, f"{cam_id}.jpg")
                if not os.path.exists(bg_path):
                    print(f"Skipping {video_name} (no background)")
                    continue
        
        print(f"\nProcessing {video_name}...")
        process_video(model, video_path, bg_path, output_path, args.device)
    
    print(f"\n✅ Background removal complete. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
