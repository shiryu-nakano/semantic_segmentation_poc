from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import numpy as np
from pathlib import Path
from utils import process_video

class HuggingFaceSegmentation:
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024"):
        print(f"Loading model: {model_name}")
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        
        # GPU利用可能ならGPUへ
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        print(f"Device: {self.device}")
    
    def inference(self, frame):
        """1フレームの推論"""
        # BGR -> RGB
        rgb_frame = frame[:, :, ::-1]
        
        # 前処理
        inputs = self.processor(images=Image.fromarray(rgb_frame), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推論
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # 後処理
        seg_mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        
        # 元のサイズにリサイズ
        import cv2
        seg_mask = cv2.resize(seg_mask, (frame.shape[1], frame.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
        
        return seg_mask

def main():
    # モデル初期化
    seg_model = HuggingFaceSegmentation()
    
    # 動画処理
    video_path = Path("videos/5750805-hd_1920_1080_24fps.mp4")
    output_path = Path("outputs/method1_output.mp4")
    output_path.parent.mkdir(exist_ok=True)
    
    avg_fps = process_video(
        video_path, 
        seg_model.inference, 
        output_path,
        display=True
    )
    
    print(f"\n=== Method 1 (Hugging Face SegFormer) ===")
    print(f"Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    main()
