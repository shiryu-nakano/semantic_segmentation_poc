import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from mmseg.apis import init_model, inference_model, show_result_pyplot
import numpy as np
from utils.video import process_video
import cv2

class MMSegmentation:
    def __init__(self, config_name='bisenetv2', checkpoint=None):
        """
        利用可能なモデル:
        - bisenetv2: 高速
        - segformer: バランス型
        - pspnet: 高精度
        """
        print(f"Loading MMSeg model: {config_name}")
        
        # モデル設定マッピング
        model_configs = {
            'bisenetv2': {
                'config': 'configs/bisenetv2/bisenetv2_fcn_4xb4-160k_cityscapes-1024x1024.py',
                'checkpoint': 'checkpoints/bisenetv2_fcn_4xb4-160k_cityscapes-1024x1024.pth'
            },
            'segformer': {
                'config': 'configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py',
                'checkpoint': 'checkpoints/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.pth'
            }
        }
        
        config_info = model_configs.get(config_name)
        config_file = checkpoint or config_info['config']
        checkpoint_file = config_info['checkpoint']
        
        # モデル初期化
        self.model = init_model(config_file, checkpoint_file, device='cuda:0')
        print(f"Model loaded successfully")
    
    def inference(self, frame):
        """1フレームの推論"""
        result = inference_model(self.model, frame)
        seg_mask = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
        
        # リサイズ(必要に応じて)
        if seg_mask.shape != frame.shape[:2]:
            seg_mask = cv2.resize(seg_mask, (frame.shape[1], frame.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        
        return seg_mask

def main():
    # モデル初期化 ('bisenetv2' or 'segformer')
    seg_model = MMSegmentation(config_name='bisenetv2')
    
    # 動画処理
    video_path = Path("videos/5750805-hd_1920_1080_24fps.mp4")
    output_path = Path("outputs/method2_output.mp4")
    output_path.parent.mkdir(exist_ok=True)
    
    avg_fps = process_video(
        video_path, 
        seg_model.inference, 
        output_path,
        display=True
    )
    
    print(f"\n=== Method 2 (MMSegmentation) ===")
    print(f"Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    main()
