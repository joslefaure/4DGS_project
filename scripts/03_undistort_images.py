#!/usr/bin/env python3
"""
Step 3: Undistort Images

Removes lens distortion from images using COLMAP calibration data.

Usage:
    python scripts/03_undistort_images.py \
        --input_dir output/frames \
        --calib_dir data/calibration \
        --output_dir output/undistorted
"""

import os
import sys
import argparse
import struct
import numpy as np
import cv2
from collections import namedtuple
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAMERA_IDS


# COLMAP camera and image structures
CameraModel = namedtuple('CameraModel', ['model_id', 'model_name', 'num_params'])
Camera = namedtuple('Camera', ['id', 'model', 'width', 'height', 'params'])
Image = namedtuple('Image', ['id', 'qvec', 'tvec', 'camera_id', 'name', 'xys', 'point3D_ids'])

CAMERA_MODELS = {
    0: CameraModel(0, 'SIMPLE_PINHOLE', 3),
    1: CameraModel(1, 'PINHOLE', 4),
    2: CameraModel(2, 'SIMPLE_RADIAL', 4),
    3: CameraModel(3, 'RADIAL', 5),
    4: CameraModel(4, 'OPENCV', 8),
    5: CameraModel(5, 'OPENCV_FISHEYE', 8),
    6: CameraModel(6, 'FULL_OPENCV', 12),
}


def read_cameras_binary(path):
    """Read COLMAP cameras.bin file."""
    cameras = {}
    with open(path, 'rb') as f:
        num_cameras = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack('<I', f.read(4))[0]
            model_id = struct.unpack('<i', f.read(4))[0]
            width = struct.unpack('<Q', f.read(8))[0]
            height = struct.unpack('<Q', f.read(8))[0]
            num_params = CAMERA_MODELS[model_id].num_params
            params = struct.unpack(f'<{num_params}d', f.read(8 * num_params))
            cameras[camera_id] = Camera(
                id=camera_id,
                model=CAMERA_MODELS[model_id].model_name,
                width=width,
                height=height,
                params=np.array(params)
            )
    return cameras


def read_images_binary(path):
    """Read COLMAP images.bin file."""
    images = {}
    with open(path, 'rb') as f:
        num_images = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack('<I', f.read(4))[0]
            qvec = struct.unpack('<4d', f.read(32))
            tvec = struct.unpack('<3d', f.read(24))
            camera_id = struct.unpack('<I', f.read(4))[0]
            
            name = ''
            while True:
                char = f.read(1).decode('utf-8')
                if char == '\x00':
                    break
                name += char
            
            num_points2D = struct.unpack('<Q', f.read(8))[0]
            f.read(24 * num_points2D)  # Skip 2D points
            
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=name,
                xys=None, point3D_ids=None
            )
    return images


def get_undistort_params(camera):
    """Get OpenCV undistortion parameters from COLMAP camera."""
    params = camera.params
    
    if camera.model == 'OPENCV':
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        k1, k2, p1, p2 = params[4], params[5], params[6], params[7]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist = np.array([k1, k2, p1, p2])
    elif camera.model == 'PINHOLE':
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist = None
    elif camera.model == 'SIMPLE_RADIAL':
        f, cx, cy, k = params[0], params[1], params[2], params[3]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        dist = np.array([k, 0, 0, 0])
    else:
        raise ValueError(f"Unsupported camera model: {camera.model}")
    
    return K, dist


def undistort_image(image_path, K, dist, output_path):
    """Undistort a single image."""
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    if dist is not None:
        img = cv2.undistort(img, K, dist)
    
    cv2.imwrite(output_path, img)
    return True


def main():
    parser = argparse.ArgumentParser(description="Undistort images using COLMAP calibration")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input frames")
    parser.add_argument("--calib_dir", type=str, required=True, help="Directory with COLMAP calibration")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    
    # Load calibration
    cameras_path = os.path.join(args.calib_dir, "cameras.bin")
    images_path = os.path.join(args.calib_dir, "images.bin")
    
    print("Loading calibration data...")
    cameras = read_cameras_binary(cameras_path)
    images = read_images_binary(images_path)
    
    # Build camera_id lookup by image name
    name_to_camera = {}
    for img in images.values():
        cam_id = img.name.split('.')[0][:3]  # Extract camera ID from name
        name_to_camera[cam_id] = cameras[img.camera_id]
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each frame directory
    frame_dirs = sorted([d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))])
    
    for frame_dir in tqdm(frame_dirs, desc="Processing frames"):
        frame_input_dir = os.path.join(args.input_dir, frame_dir)
        frame_output_dir = os.path.join(args.output_dir, frame_dir)
        os.makedirs(frame_output_dir, exist_ok=True)
        
        # Process each camera
        for cam_id in CAMERA_IDS:
            # Find image file
            for ext in ['.png', '.jpg']:
                input_path = os.path.join(frame_input_dir, f"{cam_id}{ext}")
                if os.path.exists(input_path):
                    break
            else:
                continue
            
            # Get camera calibration
            if cam_id not in name_to_camera:
                continue
            
            camera = name_to_camera[cam_id]
            K, dist = get_undistort_params(camera)
            
            output_path = os.path.join(frame_output_dir, f"{cam_id}.png")
            undistort_image(input_path, K, dist, output_path)
    
    print(f"\n✅ Undistortion complete. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
