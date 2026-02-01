import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def create_color_map(num_classes=21):
    """セマンティックセグメンテーション用カラーマップ生成"""
    colors = [
        [0, 0, 0],        # background
        [128, 64, 128],   # road
        [244, 35, 232],   # sidewalk
        [70, 70, 70],     # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],   # traffic light
        [220, 220, 0],    # traffic sign
        [107, 142, 35],   # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],   # sky
        [220, 20, 60],    # person
        [255, 0, 0],      # rider
        [0, 0, 142],      # car
        [0, 0, 70],       # truck
        [0, 60, 100],     # bus
        [0, 80, 100],     # train
        [0, 0, 230],      # motorcycle
        [119, 11, 32],    # bicycle
        [128, 128, 128],  # void/other
    ]
    
    # 追加のクラスが必要な場合はランダムカラー生成
    while len(colors) < num_classes:
        colors.append(list(np.random.randint(0, 255, 3)))
    
    return np.array(colors[:num_classes], dtype=np.uint8)

def visualize_segmentation(image, seg_mask, alpha=0.5):
    """セグメンテーション結果を可視化"""
    # seg_maskの最大値に基づいてクラス数を決定
    max_class = int(seg_mask.max())
    num_classes = max(max_class + 1, 21)
    
    color_map = create_color_map(num_classes)
    
    # 範囲外の値をクリップ（安全策）
    seg_mask = np.clip(seg_mask, 0, num_classes - 1).astype(np.uint8)
    
    colored_mask = color_map[seg_mask]
    overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    return overlay

def process_video(video_path, model_inference_fn, output_path, display=True):
    """動画処理の共通関数"""
    cap = cv2.VideoCapture(str(video_path))
    
    # 動画情報取得
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 出力動画設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    fps_list = []
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 推論実行
        import time
        start = time.time()
        seg_mask = model_inference_fn(frame)
        inference_time = time.time() - start
        fps_list.append(1.0 / inference_time)
        
        # 可視化
        result = visualize_segmentation(frame, seg_mask)
        
        # FPS表示
        cv2.putText(result, f"FPS: {fps_list[-1]:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(result)
        
        if display:
            cv2.imshow('Segmentation', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames, Avg FPS: {np.mean(fps_list):.2f}")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nAverage FPS: {np.mean(fps_list):.2f}")
    print(f"Output saved to: {output_path}")
    
    return np.mean(fps_list)
