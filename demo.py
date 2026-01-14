#!/usr/bin/env python3
"""
Create Demo Video

Renders frames from trained models and combines into a video demo.

Usage:
    python demo.py \
        --checkpoint_dir output/checkpoints \
        --data_dir output/processed \
        --output demo.mp4
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import V3_PATH, GROUP_SIZE

sys.path.insert(0, V3_PATH)


def render_frame(checkpoint_dir, data_dir, frame_idx, camera_idx=0):
    """Render a single frame from trained model."""
    
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from arguments import ModelParams, PipelineParams
    from argparse import ArgumentParser
    
    # Load model
    gaussians = GaussianModel(sh_degree=0)
    
    ckpt_path = os.path.join(checkpoint_dir, str(frame_idx))
    data_path = os.path.join(data_dir, str(frame_idx))
    
    if not os.path.exists(ckpt_path):
        return None
    
    # Find latest iteration
    pc_dir = os.path.join(ckpt_path, "point_cloud")
    if not os.path.exists(pc_dir):
        return None
    
    iterations = sorted([int(d.split("_")[-1]) for d in os.listdir(pc_dir) if d.startswith("iteration_")])
    if not iterations:
        return None
    
    latest_iter = iterations[-1]
    ply_path = os.path.join(pc_dir, f"iteration_{latest_iter}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    
    # Setup scene
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(["-s", data_path, "-m", ckpt_path, "--white_background"])
    model_args = lp.extract(args)
    pipe = pp.extract(args)
    
    scene = Scene(model_args, gaussians, load_iteration=latest_iter, shuffle=False)
    
    # Render
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    cameras = scene.getTrainCameras()
    
    if camera_idx >= len(cameras):
        camera_idx = 0
    
    out = render(cameras[camera_idx], gaussians, pipe, bg)
    rendered = out["render"].detach().cpu().numpy().transpose(1, 2, 0)
    rendered = np.clip(rendered * 255, 0, 255).astype(np.uint8)
    
    return rendered


def create_demo_video(checkpoint_dir, data_dir, output_path, fps=10, group_size=GROUP_SIZE):
    """Create demo video from rendered frames."""
    import imageio
    
    # Find available frames
    frames = []
    frame_indices = sorted([int(d) for d in os.listdir(checkpoint_dir) if d.isdigit()])
    
    print(f"Found {len(frame_indices)} trained frames")
    
    # Render each frame
    for frame_idx in tqdm(frame_indices, desc="Rendering frames"):
        rendered = render_frame(checkpoint_dir, data_dir, frame_idx)
        if rendered is not None:
            frames.append(rendered)
    
    if not frames:
        print("No frames rendered!")
        return
    
    # Save video
    print(f"Saving video to {output_path}")
    imageio.mimwrite(output_path, frames, fps=fps, quality=8)
    
    print(f"✅ Demo video saved: {output_path}")
    print(f"   Frames: {len(frames)}")
    print(f"   FPS: {fps}")
    print(f"   Duration: {len(frames)/fps:.1f}s")


def create_multi_view_demo(checkpoint_dir, data_dir, output_path, frame_idx=0, fps=5):
    """Create demo showing multiple camera views of single frame."""
    import imageio
    
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from arguments import ModelParams, PipelineParams
    from argparse import ArgumentParser
    
    ckpt_path = os.path.join(checkpoint_dir, str(frame_idx))
    data_path = os.path.join(data_dir, str(frame_idx))
    
    # Load model
    gaussians = GaussianModel(sh_degree=0)
    pc_dir = os.path.join(ckpt_path, "point_cloud")
    iterations = sorted([int(d.split("_")[-1]) for d in os.listdir(pc_dir) if d.startswith("iteration_")])
    latest_iter = iterations[-1]
    
    ply_path = os.path.join(pc_dir, f"iteration_{latest_iter}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    
    # Setup
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(["-s", data_path, "-m", ckpt_path, "--white_background"])
    model_args = lp.extract(args)
    pipe = pp.extract(args)
    
    scene = Scene(model_args, gaussians, load_iteration=latest_iter, shuffle=False)
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    cameras = scene.getTrainCameras()
    
    # Render from all cameras
    frames = []
    for cam in tqdm(cameras, desc="Rendering viewpoints"):
        out = render(cam, gaussians, pipe, bg)
        rendered = out["render"].detach().cpu().numpy().transpose(1, 2, 0)
        rendered = np.clip(rendered * 255, 0, 255).astype(np.uint8)
        frames.append(rendered)
    
    # Save video
    imageio.mimwrite(output_path, frames, fps=fps, quality=8)
    
    print(f"✅ Multi-view demo saved: {output_path}")
    print(f"   Views: {len(frames)}")


def main():
    parser = argparse.ArgumentParser(description="Create demo video")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--output", type=str, default="demo.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS")
    parser.add_argument("--multi_view", action="store_true", help="Create multi-view demo")
    parser.add_argument("--frame", type=int, default=0, help="Frame for multi-view demo")
    args = parser.parse_args()
    
    if args.multi_view:
        create_multi_view_demo(args.checkpoint_dir, args.data_dir, args.output, args.frame, args.fps)
    else:
        create_demo_video(args.checkpoint_dir, args.data_dir, args.output, args.fps)


if __name__ == "__main__":
    main()
