#!/usr/bin/env python3
"""
Step 5: Render Results

Renders novel view images from trained 4DGS model for visualization and figures.

Usage:
    python scripts/05_render_results.py \
        --checkpoint output/checkpoints/0 \
        --data_dir output/processed/0 \
        --output_dir output/renders
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import V3_PATH

# Add V3 to path
sys.path.insert(0, V3_PATH)


def render_novel_views(checkpoint_dir, data_dir, output_dir, num_views=5):
    """Render novel views from trained model."""
    
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from arguments import ModelParams, PipelineParams
    from argparse import ArgumentParser
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    gaussians = GaussianModel(sh_degree=0)
    
    # Find latest iteration
    pc_dir = os.path.join(checkpoint_dir, "point_cloud")
    iterations = sorted([int(d.split("_")[-1]) for d in os.listdir(pc_dir) if d.startswith("iteration_")])
    latest_iter = iterations[-1] if iterations else 16000
    
    ply_path = os.path.join(pc_dir, f"iteration_{latest_iter}", "point_cloud.ply")
    print(f"Loading: {ply_path}")
    gaussians.load_ply(ply_path)
    
    # Setup scene
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(["-s", data_dir, "-m", checkpoint_dir, "--white_background"])
    model_args = lp.extract(args)
    pipe = pp.extract(args)
    
    scene = Scene(model_args, gaussians, load_iteration=latest_iter, shuffle=False)
    
    # Render
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    cameras = scene.getTrainCameras()
    
    print(f"Rendering {min(num_views, len(cameras))} views...")
    
    for i in range(min(num_views, len(cameras))):
        cam = cameras[i]
        
        # Render
        out = render(cam, gaussians, pipe, bg)
        rendered = out["render"].detach().cpu().numpy().transpose(1, 2, 0)
        rendered = np.clip(rendered * 255, 0, 255).astype(np.uint8)
        
        # Ground truth
        gt = cam.original_image.cpu().numpy().transpose(1, 2, 0)
        gt = np.clip(gt * 255, 0, 255).astype(np.uint8)
        
        # Save
        Image.fromarray(gt).save(os.path.join(output_dir, f"gt_{i:03d}.png"))
        Image.fromarray(rendered).save(os.path.join(output_dir, f"rendered_{i:03d}.png"))
        
        print(f"  Saved view {i}")
    
    print(f"\n✅ Renders saved to: {output_dir}")


def create_comparison_figure(checkpoint_dir, data_dir, output_path, input_cam=0, novel_cam=10):
    """Create side-by-side comparison figure (input vs novel view)."""
    
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from arguments import ModelParams, PipelineParams
    from argparse import ArgumentParser
    
    # Load model
    gaussians = GaussianModel(sh_degree=0)
    
    pc_dir = os.path.join(checkpoint_dir, "point_cloud")
    iterations = sorted([int(d.split("_")[-1]) for d in os.listdir(pc_dir) if d.startswith("iteration_")])
    latest_iter = iterations[-1] if iterations else 16000
    
    ply_path = os.path.join(pc_dir, f"iteration_{latest_iter}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    
    # Setup
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(["-s", data_dir, "-m", checkpoint_dir, "--white_background"])
    model_args = lp.extract(args)
    pipe = pp.extract(args)
    
    scene = Scene(model_args, gaussians, load_iteration=latest_iter, shuffle=False)
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    cameras = scene.getTrainCameras()
    
    # Get input (GT from one camera)
    gt = cameras[input_cam].original_image.cpu().numpy().transpose(1, 2, 0)
    gt = np.clip(gt * 255, 0, 255).astype(np.uint8)
    
    # Render from different camera (novel view)
    out = render(cameras[novel_cam], gaussians, pipe, bg)
    rendered = out["render"].detach().cpu().numpy().transpose(1, 2, 0)
    rendered = np.clip(rendered * 255, 0, 255).astype(np.uint8)
    
    # Resize to same height
    h = min(gt.shape[0], rendered.shape[0])
    w1 = int(gt.shape[1] * h / gt.shape[0])
    w2 = int(rendered.shape[1] * h / rendered.shape[0])
    
    gt_resized = np.array(Image.fromarray(gt).resize((w1, h), Image.LANCZOS))
    rendered_resized = np.array(Image.fromarray(rendered).resize((w2, h), Image.LANCZOS))
    
    # Combine with gap
    gap = np.ones((h, 20, 3), dtype=np.uint8) * 255
    comparison = np.concatenate([gt_resized, gap, rendered_resized], axis=1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(comparison).save(output_path)
    
    print(f"✅ Saved comparison figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Render results from trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_views", type=int, default=5, help="Number of views to render")
    parser.add_argument("--figure", action="store_true", help="Create comparison figure")
    parser.add_argument("--input_cam", type=int, default=8, help="Input camera index")
    parser.add_argument("--novel_cam", type=int, default=15, help="Novel view camera index")
    args = parser.parse_args()
    
    if args.figure:
        output_path = os.path.join(args.output_dir, "figure_comparison.png")
        create_comparison_figure(args.checkpoint, args.data_dir, output_path, args.input_cam, args.novel_cam)
    else:
        render_novel_views(args.checkpoint, args.data_dir, args.output_dir, args.num_views)


if __name__ == "__main__":
    main()
