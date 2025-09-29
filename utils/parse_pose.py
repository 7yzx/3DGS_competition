import cv2
import os
import time
import numpy as np
import time
from read_write_model import rotmat2qvec, qvec2rotmat, Camera, write_cameras_text, write_images_text, write_points3D_text, Image

from video import parse_video, parse_video2

import open3d as o3d

def _parse_cameras(cam_file):
    """解析 camera.txt: 相机内参和畸变参数"""
    cameras = {}
    if not os.path.exists(cam_file):
        print(f"⚠️ {cam_file} 不存在")
        return
    
    with open(cam_file, "r") as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 11:
                continue
            cam_id = int(parts[0])
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            fx, fy, cx, cy = map(float, parts[4:8])
            dist = list(map(float, parts[8:]))  # 畸变系数
            cameras[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "params": np.array([fx, fy, cx, cy], dtype=np.float32),
                "distortion": dist
            }
    return cameras

def _parse_images(img_file):
    """解析 images.txt: 位姿（四元数+平移+时间戳+矩阵）"""
    if not os.path.exists(img_file):
        print(f"⚠️ {img_file} 不存在")
        return
    images = {}

    with open(img_file, "r") as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            cam_id = int(parts[8])
            img_name = parts[9]
            R = np.transpose(qvec2rotmat(np.array([qw, qx, qy, qz])))
            T = np.array([tx, ty, tz])
            mat4x4 = np.eye(4)
            mat4x4[:3, :3] = R
            mat4x4[:3, 3] = T
            images[image_id] = {
                "qvec": np.array([qw, qx, qy, qz]),
                "tvec": np.array([tx, ty, tz]),
                "cam_id": cam_id,
                "name": img_name,
                "mat": mat4x4
            }
    return images

def _parse_points3D(pts_file):
    """解析 points3D.txt: 三维点云"""
    if not os.path.exists(pts_file):
        print(f"⚠️ {pts_file} 不存在")
        return
    points3D = {}

    with open(pts_file, "r") as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            pid = int(parts[0])
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            error = float(parts[7])
            points3D[pid] = {
                "id": pid,
                "xyz": np.array([x, y, z]),
                "rgb": (r, g, b),
                "error": error
            }
    return points3D



def save_to_colmap(camera_o, images_o, points3D_o, sparse_dir):

    cameras = {}
    width = camera_o[0]["width"]
    height = camera_o[0]["height"]
    params = camera_o[0]["params"]
    camera_id = 1 
    cameras[camera_id] = Camera(
                    id=camera_id,
                    model="PINHOLE",
                    width=width,
                    height=height,
                    params=params,
                )
    camera_txt_path = os.path.join(sparse_dir, "cameras.txt")
    write_cameras_text(cameras, camera_txt_path)
    
    images = {}
    
    for i, img in enumerate(images_o.values()):
        image = Image(
            id=i+1,
            qvec=img["qvec"],
            tvec=img["tvec"],
            camera_id=1,
            name=img["name"],
            xys=np.zeros((0, 2)),
            point3D_ids=np.array([], dtype=int)
        )
        images[i+1] = image

    images_txt_path = os.path.join(sparse_dir, "images.txt")
    write_images_text(images, images_txt_path)
            
    # # 创建points3D.txt
    # points3D_txt_path = os.path.join(sparse_dir, "points3D.txt")
    # write_points3D_text(points3D_o, points3D_txt_path)
    
    


def save_points3D_to_ply(points3D, ply_path):
    # 提取坐标和颜色
    coords = []
    colors = []
    for pt in points3D.values():
        coords.append(pt["xyz"])
        colors.append(np.array(pt["rgb"]) / 255.0)  # Open3D 颜色范围 [0,1]

    coords = np.array(coords)
    colors = np.array(colors)

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 保存为 ply
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved {len(coords)} points to {ply_path}")

def parse_slam(slam_dir):
    
    cam_file = os.path.join(slam_dir, "cameras.txt")
    img_file = os.path.join(slam_dir, "images.txt")
    pts_file = os.path.join(slam_dir, "points3D.txt")
    
    camera = _parse_cameras(cam_file)
    images = _parse_images(img_file)
    points3D = _parse_points3D(pts_file)

    return camera, images, points3D


if __name__ == "__main__":
    base_dir = "/home/farsee2/YZX_code/datasets/Final/"
    dataset_names = sorted([f for f in os.listdir(base_dir)])
    test_num = 10
    for i, name in enumerate(dataset_names):
        # if i == test_num:
        #     break

        start_time = time.time()
        dataset_path = os.path.join(base_dir, name)
        video_path = os.path.join(dataset_path, f"{name}_flip.mp4")
        video_info_path = os.path.join(dataset_path, "inputs", "videoInfo.txt")
        # output_dir = os.path.join(dataset_path, "frames")
        output_dir = dataset_path
        sparse_dir = os.path.join(output_dir, "sparse", "0")
        os.makedirs(sparse_dir, exist_ok=True)
        slam_dir = os.path.join(dataset_path, "inputs", "slam")

        camera_o, images_o, points3D_o = parse_slam(slam_dir)
        imagelist = [img["name"] for img in images_o.values()]
        parse_video2(video_path, video_info_path, output_dir, camera_o, imagelist)
        
        save_to_colmap(camera_o, images_o, points3D_o, sparse_dir)
        
        end_time = time.time()
        
        save_points3D_to_ply(points3D_o, os.path.join(sparse_dir, "points3D.ply"))
        print(f"parse slam file: {name}, 用时: {end_time - start_time:.2f} 秒")
        
        