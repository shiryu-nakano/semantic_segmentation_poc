# compare_methods.py
from pathlib import Path
import json
import time
import argparse
from utils import process_video

def compare_all_methods(video_path, output_dir='outputs', display=False):
    """全手法を比較実行"""
    results = {}
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    print("="*80)
    print(f"Comparing Semantic Segmentation Methods")
    print(f"Video: {video_path}")
    print("="*80)
    
    # Method 1: Hugging Face SegFormer-B0
    print("\n" + "="*80)
    print("Method 1: Hugging Face SegFormer-B0")
    print("="*80)
    try:
        from method1_huggingface import HuggingFaceSegmentation
        
        model1 = HuggingFaceSegmentation()
        fps1 = process_video(
            video_path,
            model1.inference,
            output_dir / "method1_huggingface_output.mp4",
            display=display
        )
        results['method1_huggingface_segformer_b0'] = {
            'fps': fps1,
            'model': 'SegFormer-B0',
            'framework': 'Hugging Face Transformers',
            'success': True
        }
    except Exception as e:
        print(f"Method 1 failed: {e}")
        results['method1_huggingface_segformer_b0'] = {
            'fps': 0,
            'error': str(e),
            'success': False
        }
    
    # Method 2: SegFormer-B1 Direct
    print("\n" + "="*80)
    print("Method 2: SegFormer-B1 Direct")
    print("="*80)
    try:
        from method2_segformer_direct import SegFormerDirect
        
        model2 = SegFormerDirect(model_size='b1')
        fps2 = process_video(
            video_path,
            model2.inference,
            output_dir / "method2_segformer_b1_output.mp4",
            display=display
        )
        results['method2_segformer_b1'] = {
            'fps': fps2,
            'model': 'SegFormer-B1',
            'framework': 'Hugging Face Transformers',
            'success': True
        }
    except Exception as e:
        print(f"Method 2 failed: {e}")
        results['method2_segformer_b1'] = {
            'fps': 0,
            'error': str(e),
            'success': False
        }
    
    # Method 3: TorchVision DeepLabV3
    print("\n" + "="*80)
    print("Method 3: TorchVision DeepLabV3 + MobileNetV3")
    print("="*80)
    try:
        from method3_torchvision import TorchVisionSegmentation
        
        model3 = TorchVisionSegmentation()
        fps3 = process_video(
            video_path,
            model3.inference,
            output_dir / "method3_torchvision_output.mp4",
            display=display
        )
        results['method3_torchvision_deeplabv3'] = {
            'fps': fps3,
            'model': 'DeepLabV3 + MobileNetV3',
            'framework': 'TorchVision',
            'success': True
        }
    except Exception as e:
        print(f"Method 3 failed: {e}")
        results['method3_torchvision_deeplabv3'] = {
            'fps': 0,
            'error': str(e),
            'success': False
        }
    
    # 結果表示
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    # 表形式で表示
    print(f"\n{'Method':<40} {'Model':<30} {'FPS':>10} {'Status':>10}")
    print("-" * 90)
    
    for method_name, info in results.items():
        status = "✓ OK" if info['success'] else "✗ FAIL"
        model_name = info.get('model', 'N/A')
        fps = info.get('fps', 0)
        print(f"{method_name:<40} {model_name:<30} {fps:>10.2f} {status:>10}")
    
    # 最速の手法を表示
    successful_results = {k: v for k, v in results.items() if v['success']}
    if successful_results:
        fastest = max(successful_results.items(), key=lambda x: x[1]['fps'])
        print(f"\n🏆 Fastest method: {fastest[0]} ({fastest[1]['fps']:.2f} FPS)")
    
    # JSON形式で保存
    results_file = output_dir / 'comparison_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Markdown形式でも保存
    markdown_file = output_dir / 'comparison_results.md'
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write("# Semantic Segmentation Methods Comparison\n\n")
        f.write(f"**Video:** `{video_path}`\n\n")
        f.write("## Results\n\n")
        f.write("| Method | Model | Framework | FPS | Status |\n")
        f.write("|--------|-------|-----------|-----|--------|\n")
        
        for method_name, info in results.items():
            status = "✓" if info['success'] else "✗"
            model_name = info.get('model', 'N/A')
            framework = info.get('framework', 'N/A')
            fps = info.get('fps', 0)
            f.write(f"| {method_name} | {model_name} | {framework} | {fps:.2f} | {status} |\n")
        
        if successful_results:
            f.write(f"\n**Fastest:** {fastest[0]} ({fastest[1]['fps']:.2f} FPS)\n")
    
    print(f"Markdown report saved to: {markdown_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Compare all semantic segmentation methods')
    parser.add_argument('--video', '-v', type=str,
                       default='videos/5750805-hd_1920_1080_24fps.mp4',
                       help='Input video path')
    parser.add_argument('--output-dir', '-o', type=str,
                       default='outputs',
                       help='Output directory')
    parser.add_argument('--display', '-d', action='store_true',
                       help='Display video during processing')
    
    args = parser.parse_args()
    
    # 利用可能な動画ファイルを表示
    videos_dir = Path("videos")
    if not Path(args.video).exists() and videos_dir.exists():
        print("Available videos:")
        for f in videos_dir.glob("*.mp4"):
            print(f"  - {f}")
        print(f"\nSpecified video not found: {args.video}")
        print("Please specify a valid video with --video option")
        return
    
    compare_all_methods(args.video, args.output_dir, args.display)

if __name__ == "__main__":
    main()
