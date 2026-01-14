#!/usr/bin/env python3
"""
4DGS Training Script

Trains 4D Gaussian Splatting model on prepared multi-view data.

Usage:
    python train.py \
        --data_dir output/processed \
        --output_dir output/checkpoints \
        --iterations 16000

For sequence training (multiple frames):
    python train.py \
        --data_dir output/processed \
        --output_dir output/checkpoints \
        --sequence \
        --start_frame 0 \
        --end_frame 300 \
        --group_size 20
"""

import os
import sys
import argparse
import subprocess

from config import V3_PATH, ITERATIONS, GROUP_SIZE, WHITE_BACKGROUND


def train_single_frame(data_dir, output_dir, frame_idx, iterations=ITERATIONS):
    """Train model for a single frame."""
    
    source_path = os.path.join(data_dir, str(frame_idx))
    model_path = os.path.join(output_dir, str(frame_idx))
    
    cmd = [
        sys.executable,
        os.path.join(V3_PATH, "train.py"),
        "-s", source_path,
        "-m", model_path,
        "--iterations", str(iterations),
    ]
    
    if WHITE_BACKGROUND:
        cmd.append("--white_background")
    
    print(f"\n{'='*60}")
    print(f"Training frame {frame_idx}")
    print(f"Source: {source_path}")
    print(f"Output: {model_path}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=V3_PATH)
    return result.returncode == 0


def train_sequence(data_dir, output_dir, start_frame, end_frame, group_size=GROUP_SIZE, iterations=ITERATIONS):
    """Train sequence with grouped dynamic training."""
    
    # Train head frames (every group_size frames)
    for frame_idx in range(start_frame, end_frame, group_size):
        print(f"\n{'='*60}")
        print(f"Training head frame {frame_idx} (group size: {group_size})")
        print(f"{'='*60}\n")
        
        success = train_single_frame(data_dir, output_dir, frame_idx, iterations)
        
        if not success:
            print(f"Warning: Training failed for frame {frame_idx}")
    
    print(f"\n✅ Sequence training complete!")
    print(f"Head frames: {list(range(start_frame, end_frame, group_size))}")


def main():
    parser = argparse.ArgumentParser(description="Train 4DGS model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with prepared data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--iterations", type=int, default=ITERATIONS, help="Training iterations")
    parser.add_argument("--sequence", action="store_true", help="Train sequence (multiple frames)")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--end_frame", type=int, default=300, help="End frame index")
    parser.add_argument("--group_size", type=int, default=GROUP_SIZE, help="Frames per group")
    parser.add_argument("--frame", type=int, default=None, help="Single frame to train")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.frame is not None:
        # Train single frame
        train_single_frame(args.data_dir, args.output_dir, args.frame, args.iterations)
    elif args.sequence:
        # Train sequence
        train_sequence(
            args.data_dir, args.output_dir,
            args.start_frame, args.end_frame,
            args.group_size, args.iterations
        )
    else:
        # Default: train frame 0
        train_single_frame(args.data_dir, args.output_dir, 0, args.iterations)


if __name__ == "__main__":
    main()
