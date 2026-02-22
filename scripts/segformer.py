import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import numpy as np
from utils.video import process_video
import cv2
import argparse


class SegFormerSegmentation:
    def __init__(self, model_size='b0'):
        model_name = f"nvidia/segformer-{model_size}-finetuned-cityscapes-1024-1024"
        print(f"Loading model: {model_name}")

        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        print(f"Device: {self.device}")

    def inference(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=Image.fromarray(rgb_frame), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        seg_mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        seg_mask = cv2.resize(seg_mask, (frame.shape[1], frame.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        return seg_mask


def main():
    parser = argparse.ArgumentParser(description='SegFormer semantic segmentation')
    parser.add_argument('video', type=str,
                        help='Video path relative to videos/ directory (e.g., tsukuba2025/clip.mp4)')
    parser.add_argument('--model-size', '-m', type=str,
                        default='b0',
                        choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
                        help='SegFormer model size (default: b0)')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable display window')

    args = parser.parse_args()

    video_path = Path("videos") / args.video
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return

    output_path = Path("out") / args.video
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seg_model = SegFormerSegmentation(model_size=args.model_size)

    avg_fps = process_video(
        video_path,
        seg_model.inference,
        output_path,
        display=not args.no_display
    )

    print(f"\n=== SegFormer-{args.model_size.upper()} ===")
    print(f"Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()
