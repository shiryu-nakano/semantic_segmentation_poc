from pathlib import Path
import json

def compare_all_methods():
    results = {}
    
    # Method 1
    print("\n" + "="*50)
    print("Testing Method 1: Hugging Face SegFormer")
    print("="*50)
    from method1_huggingface import HuggingFaceSegmentation
    from utils import process_video
    
    model1 = HuggingFaceSegmentation()
    fps1 = process_video(
        Path("videos/test_video.mp4"),
        model1.inference,
        Path("outputs/method1_output.mp4"),
        display=False
    )
    results['method1_huggingface'] = fps1
    
    # Method 2
    print("\n" + "="*50)
    print("Testing Method 2: MMSegmentation")
    print("="*50)
    from method2_mmseg import MMSegmentation
    
    model2 = MMSegmentation(config_name='bisenetv2')
    fps2 = process_video(
        Path("videos/test_video.mp4"),
        model2.inference,
        Path("outputs/method2_output.mp4"),
        display=False
    )
    results['method2_mmseg'] = fps2
    
    # Method 3
    print("\n" + "="*50)
    print("Testing Method 3: TorchVision")
    print("="*50)
    from method3_torchvision import TorchVisionSegmentation
    
    model3 = TorchVisionSegmentation()
    fps3 = process_video(
        Path("videos/test_video.mp4"),
        model3.inference,
        Path("outputs/method3_output.mp4"),
        display=False
    )
    results['method3_torchvision'] = fps3
    
    # 結果表示
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    for method, fps in results.items():
        print(f"{method}: {fps:.2f} FPS")
    
    # 結果保存
    with open('outputs/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    compare_all_methods()
