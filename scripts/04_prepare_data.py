#!/usr/bin/env python3
"""
Step 4: Prepare Training Data

Formats data for V3 4DGS training by creating transforms.json files.

Usage:
    python scripts/04_prepare_data.py \
        --input_dir output/undistorted \
        --calib_dir data/calibration \
        --output_dir output/processed
"""

import os
import sys
import argparse
import struct
import json
import numpy as np
from collections import namedtuple
from tqdm import tqdm
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAMERA_IDS, NUM_FRAMES


# COLMAP structures
CameraModel = namedtuple('CameraModel', ['model_id', 'model_name', 'num_params'])
Camera = namedtuple('Camera', ['id', 'model', 'width', 'height', 'params'])
Point3D = namedtuple('Point3D', ['id', 'xyz', 'rgb', 'error', 'image_ids', 'point2D_idxs'])
Image = namedtuple('Image', ['id', 'qvec', 'tvec', 'camera_id', 'name', 'xys', 'point3D_ids'])

CAMERA_MODELS = {
    0: CameraModel(0, 'SIMPLE_PINHOLE', 3),
    1: CameraModel(1, 'PINHOLE', 4),
    2: CameraModel(2, 'SIMPLE_RADIAL', 4),
    3: CameraModel(3, 'RADIAL', 5),
    4: CameraModel(4, 'OPENCV', 8),
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
            f.read(24 * num_points2D)
            
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=name,
                xys=None, point3D_ids=None
            )
    return images


def read_points3D_binary(path):
    """Read COLMAP points3D.bin file."""
    points3D = {}
    with open(path, 'rb') as f:
        num_points = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_points):
            point_id = struct.unpack('<Q', f.read(8))[0]
            xyz = struct.unpack('<3d', f.read(24))
            rgb = struct.unpack('<3B', f.read(3))
            error = struct.unpack('<d', f.read(8))[0]
            track_length = struct.unpack('<Q', f.read(8))[0]
            image_ids = []
            point2D_idxs = []
            for _ in range(track_length):
                image_id = struct.unpack('<I', f.read(4))[0]
                point2D_idx = struct.unpack('<I', f.read(4))[0]
                image_ids.append(image_id)
                point2D_idxs.append(point2D_idx)
            points3D[point_id] = Point3D(
                id=point_id, xyz=np.array(xyz), rgb=np.array(rgb),
                error=error, image_ids=image_ids, point2D_idxs=point2D_idxs
            )
    return points3D


def write_points3D_ply(points3D, output_path):
    """Write points3D to PLY format."""
    points = list(points3D.values())
    
    header = f"""ply
format binary_little_endian 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    with open(output_path, 'wb') as f:
        f.write(header.encode('utf-8'))
        for p in points:
            f.write(struct.pack('<fff', *p.xyz.astype(np.float32)))
            f.write(struct.pack('<BBB', *p.rgb.astype(np.uint8)))


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    q = np.array(qvec)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    return R


def create_transforms_json(images, cameras, frame_images_dir, frame_idx):
    """Create transforms.json for one frame."""
    
    # Coordinate system flip (COLMAP to rendering convention)
    flip_mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    
    frames = []
    
    for img in images.values():
        cam_id = img.name.split('.')[0][:3]
        
        # Skip missing cameras
        if cam_id not in CAMERA_IDS:
            continue
        
        # Check if image exists
        img_name = f"{cam_id}001.png"
        img_path = os.path.join(frame_images_dir, img_name)
        if not os.path.exists(img_path):
            continue
        
        # Get camera intrinsics for this specific camera
        camera = cameras[img.camera_id]
        params = camera.params
        
        if camera.model == 'OPENCV':
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        elif camera.model == 'PINHOLE':
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        else:
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        
        # Compute transform matrix
        R = qvec2rotmat(img.qvec)
        t = np.array(img.tvec)
        
        # Apply flip
        R = flip_mat @ R
        t = flip_mat @ t
        
        # World-to-camera -> Camera-to-world
        R_inv = R.T
        t_inv = -R_inv @ t
        
        # 4x4 transform matrix
        transform = np.eye(4)
        transform[:3, :3] = R_inv
        transform[:3, 3] = t_inv
        
        # V3 expects camera params inside each frame
        frames.append({
            "file_path": f"images/{img_name}",
            "transform_matrix": transform.tolist(),
            "w": float(camera.width),
            "h": float(camera.height),
            "fl_x": float(fx),
            "fl_y": float(fy),
            "cx": float(cx),
            "cy": float(cy),
        })
    
    transforms = {
        "frames": frames
    }
    
    return transforms


def main():
    parser = argparse.ArgumentParser(description="Prepare data for V3 training")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with undistorted frames")
    parser.add_argument("--calib_dir", type=str, required=True, help="Directory with COLMAP calibration")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_frames", type=int, default=NUM_FRAMES, help="Number of frames")
    args = parser.parse_args()
    
    # Load calibration
    print("Loading calibration...")
    cameras = read_cameras_binary(os.path.join(args.calib_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(args.calib_dir, "images.bin"))
    
    # Load and convert points3D to PLY (V3 expects points3d.ply)
    points3d_bin = os.path.join(args.calib_dir, "points3D.bin")
    points3D = None
    if os.path.exists(points3d_bin):
        print("Loading points3D...")
        points3D = read_points3D_binary(points3d_bin)
        print(f"  Loaded {len(points3D)} 3D points")
    
    # Process each frame
    for frame_idx in tqdm(range(args.num_frames), desc="Preparing frames"):
        frame_output_dir = os.path.join(args.output_dir, str(frame_idx))
        images_output_dir = os.path.join(frame_output_dir, "images")
        sparse_output_dir = os.path.join(frame_output_dir, "sparse/0")
        
        os.makedirs(images_output_dir, exist_ok=True)
        os.makedirs(sparse_output_dir, exist_ok=True)
        
        # Find input images for this frame
        frame_input_found = False
        for cam_id in CAMERA_IDS:
            # Try different input directory structures
            possible_inputs = [
                os.path.join(args.input_dir, str(frame_idx), "images", f"{cam_id}001.png"),
                os.path.join(args.input_dir, str(frame_idx), f"{cam_id}001.png"),
                os.path.join(args.input_dir, str(frame_idx), f"{cam_id}.png"),
                os.path.join(args.input_dir, cam_id, f"{frame_idx:04d}.png"),
            ]
            
            for src_path in possible_inputs:
                if os.path.exists(src_path):
                    dst_path = os.path.join(images_output_dir, f"{cam_id}001.png")
                    shutil.copy2(src_path, dst_path)
                    frame_input_found = True
                    break
        
        if not frame_input_found:
            print(f"Warning: No images found for frame {frame_idx}")
            continue
        
        # Create transforms.json
        transforms = create_transforms_json(images, cameras, images_output_dir, frame_idx)
        
        with open(os.path.join(frame_output_dir, "transforms.json"), 'w') as f:
            json.dump(transforms, f, indent=2)
        
        # Create points3d.ply (V3 expects this file)
        if points3D is not None:
            write_points3D_ply(points3D, os.path.join(frame_output_dir, "points3d.ply"))
        
        # Copy COLMAP files for reference
        shutil.copy2(os.path.join(args.calib_dir, "cameras.bin"), os.path.join(sparse_output_dir, "cameras.bin"))
        shutil.copy2(os.path.join(args.calib_dir, "images.bin"), os.path.join(sparse_output_dir, "images.bin"))
        if os.path.exists(points3d_bin):
            shutil.copy2(points3d_bin, os.path.join(sparse_output_dir, "points3D.bin"))
    
    print(f"\n✅ Data preparation complete. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
