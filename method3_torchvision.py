import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import numpy as np
from pathlib import Path
from utils import process_video
import cv2

class TorchVisionSegmentation:
    def __init__(self):
        print("Loading TorchVision DeepLabV3 MobileNetV3")
        
        # 学習済みモデルロード
        self.model = deeplabv3_mobilenet_v3_large(pretrained=True)
        
        # GPU利用
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        
        # 正規化パラメータ (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        print(f"Device: {self.device}")
    
    def inference(self, frame):
        """1フレームの推論"""
        h, w = frame.shape[:2]
        
        # 前処理
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        input_tensor = (input_tensor - self.mean) / self.std
        
        # 推論
        with torch.no_grad():
            output = self.model(input_tensor)['out']
            seg_mask = output.argmax(1)[0].cpu().numpy().astype(np.uint8)
        
        # リサイズ
        seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return seg_mask

def main():
    # モデル初期化
    seg_model = TorchVisionSegmentation()
    
    # 動画処理
    video_path = Path("videos/5750805-hd_1920_1080_24fps.mp4")
    output_path = Path("outputs/method3_output.mp4")
    output_path.parent.mkdir(exist_ok=True)
    
    avg_fps = process_video(
        video_path, 
        seg_model.inference, 
        output_path,
        display=True
    )
    
    print(f"\n=== Method 3 (TorchVision DeepLabV3) ===")
    print(f"Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    main()
