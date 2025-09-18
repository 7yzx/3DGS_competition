import cv2
import os
import time
import numpy as np 


def parse_video(video_path, video_info_path, output_dir, imagelist=None):
    """
    从视频中提取指定帧并保存为图片，图片命名为时间戳。
    
    参数:
    video_path: 视频文件路径
    video_info_path: 包含帧ID和时间戳的文本文件路径
    output_dir: 输出图片的文件夹路径
    """
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件")

    # 读取 videoInfo.txt
    with open(video_info_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        
        frame_id = int(parts[0])
        timestamp = parts[1]  # 用作文件名
        
        # 定位到对应帧 (注意：opencv 从 0 开始计数)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ 无法读取第 {frame_id} 帧")
            continue
        if timestamp in imagelist or imagelist is None:
            # 保存图片，命名为时间戳
            out_path = os.path.join(output_dir, f"{timestamp}.jpg")
            cv2.imwrite(out_path, frame)
            # print(f"✅ 保存: {out_path}")

    cap.release()


def _parse_videoinfo(video_info_path):
    frameid_to_timestamp = {}
    with open(video_info_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                frame_id = int(parts[0])
                timestamp = parts[1]
                frameid_to_timestamp[frame_id] = timestamp
    return frameid_to_timestamp
                
def parse_video2(video_path, video_info_path, output_dir, camera, needed_images=None):
    """
    从视频中提取指定帧并保存为图片，图片命名为时间戳。
    
    参数:
    video_path: 视频文件路径
    video_info_path: 包含帧ID和时间戳的文本文件路径
    output_dir: 输出图片的文件夹路径
    """
    dist = np.array(camera["distortion"], dtype=np.float32)

    K = np.array([
        [camera["fx"], 0, camera["cx"]],
        [0, camera["fy"], camera["cy"]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    frameid_to_timestamp = _parse_videoinfo(video_info_path)
    needed_ts_set = set([img.replace(".jpg","") for img in needed_images])

    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        # 判断当前帧是否需要保存
        if frame_id in frameid_to_timestamp:
            ts = frameid_to_timestamp[frame_id]
            if ts in needed_ts_set:
                # 去畸变
                # undistorted = cv2.undistort(frame, K, dist)
                # 保存
                save_path = os.path.join(output_dir, f"{ts}.jpg")
                # cv2.imwrite(save_path, undistorted)
                cv2.imwrite(os.path.join(output_dir, f"{ts}.jpg"), frame)
                
    cap.release()


if __name__ == "__main__":
    base_dir = "/home/farsee2/YZX_code/datasets/Final/"
    dataset_names = sorted([f for f in os.listdir(base_dir)])
    test_num = 2
    for i, name in enumerate(dataset_names):
        if i == test_num:
            break

        start_time = time.time()
        dataset_path = os.path.join(base_dir, name)
        video_path = os.path.join(dataset_path, f"{name}_flip.mp4")
        video_info_path = os.path.join(dataset_path, "inputs", "videoInfo.txt")
        output_dir = os.path.join(dataset_path, "frames")

        parse_video(video_path, video_info_path, output_dir)
        
        end_time = time.time()
        
        print(f"✅ 处理完成: {name}, 用时: {end_time - start_time:.2f} 秒")
        
