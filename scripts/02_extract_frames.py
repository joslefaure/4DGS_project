#!/usr/bin/env python3
"""
Step 2: Extract Frames

Extracts individual frames from processed videos.

Usage:
    python scripts/02_extract_frames.py \
        --input_dir output/nobg \
        --output_dir output/frames \
        --num_frames 300
"""

import os
import sys
import argparse
import subprocess
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAMERA_IDS, NUM_FRAMES


def extract_frames(video_path, output_dir, num_frames=None):
    """Extract frames from a video using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-start_number", "0",
    ]
    
    if num_frames:
        cmd.extend(["-vframes", str(num_frames)])
    
    cmd.append(os.path.join(output_dir, "%04d.png"))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for frames")
    parser.add_argument("--num_frames", type=int, default=NUM_FRAMES, help="Number of frames to extract")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for cam_id in tqdm(CAMERA_IDS, desc="Extracting frames"):
        video_name = f"{cam_id}001.mp4"
        video_path = os.path.join(args.input_dir, video_name)
        
        if not os.path.exists(video_path):
            print(f"Skipping {video_name} (not found)")
            continue
        
        # Output: output_dir/CAM_ID/0000.png, 0001.png, ...
        cam_output_dir = os.path.join(args.output_dir, cam_id)
        extract_frames(video_path, cam_output_dir, args.num_frames)
    
    print(f"\n✅ Frame extraction complete. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
