import cv2
import numpy as np


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

    while len(colors) < num_classes:
        colors.append(list(np.random.randint(0, 255, 3)))

    return np.array(colors[:num_classes], dtype=np.uint8)


def visualize_segmentation(image, seg_mask, alpha=0.5):
    """セグメンテーション結果を可視化"""
    max_class = int(seg_mask.max())
    num_classes = max(max_class + 1, 21)

    color_map = create_color_map(num_classes)

    seg_mask = np.clip(seg_mask, 0, num_classes - 1).astype(np.uint8)

    colored_mask = color_map[seg_mask]
    overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    return overlay
