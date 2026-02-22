import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import numpy as np
from utils.video import process_video
import cv2
import argparse


class DeepLabV3Segmentation:
    def __init__(self):
        print("Loading TorchVision DeepLabV3 MobileNetV3")

        self.model = deeplabv3_mobilenet_v3_large(pretrained=True)

        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        # 正規化パラメータ (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        print(f"Device: {self.device}")

    def inference(self, frame):
        h, w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        input_tensor = (input_tensor - self.mean) / self.std

        with torch.no_grad():
            output = self.model(input_tensor)['out']
            seg_mask = output.argmax(1)[0].cpu().numpy().astype(np.uint8)

        seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return seg_mask


def main():
    parser = argparse.ArgumentParser(description='DeepLabV3 + MobileNetV3 semantic segmentation')
    parser.add_argument('video', type=str,
                        help='Video path relative to videos/ directory (e.g., tsukuba2025/clip.mp4)')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable display window')

    args = parser.parse_args()

    video_path = Path("videos") / args.video
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return

    output_path = Path("out") / args.video
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seg_model = DeepLabV3Segmentation()

    avg_fps = process_video(
        video_path,
        seg_model.inference,
        output_path,
        display=not args.no_display
    )

    print(f"\n=== DeepLabV3 + MobileNetV3 ===")
    print(f"Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()
