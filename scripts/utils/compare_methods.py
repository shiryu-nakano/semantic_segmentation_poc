# compare_methods.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import argparse
from utils.video import process_video

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

    # SegFormer-B0
    print("\n" + "="*80)
    print("SegFormer-B0")
    print("="*80)
    try:
        from segformer import SegFormerSegmentation

        model1 = SegFormerSegmentation(model_size='b0')
        fps1 = process_video(
            video_path,
            model1.inference,
            output_dir / "segformer_b0_output.mp4",
            display=display
        )
        results['segformer_b0'] = {
            'fps': fps1,
            'model': 'SegFormer-B0',
            'framework': 'Hugging Face Transformers',
            'success': True
        }
    except Exception as e:
        print(f"SegFormer-B0 failed: {e}")
        results['segformer_b0'] = {'fps': 0, 'error': str(e), 'success': False}

    # SegFormer-B1
    print("\n" + "="*80)
    print("SegFormer-B1")
    print("="*80)
    try:
        from segformer import SegFormerSegmentation

        model2 = SegFormerSegmentation(model_size='b1')
        fps2 = process_video(
            video_path,
            model2.inference,
            output_dir / "segformer_b1_output.mp4",
            display=display
        )
        results['segformer_b1'] = {
            'fps': fps2,
            'model': 'SegFormer-B1',
            'framework': 'Hugging Face Transformers',
            'success': True
        }
    except Exception as e:
        print(f"SegFormer-B1 failed: {e}")
        results['segformer_b1'] = {'fps': 0, 'error': str(e), 'success': False}

    # DeepLabV3 + MobileNetV3
    print("\n" + "="*80)
    print("DeepLabV3 + MobileNetV3")
    print("="*80)
    try:
        from deeplabv3 import DeepLabV3Segmentation

        model3 = DeepLabV3Segmentation()
        fps3 = process_video(
            video_path,
            model3.inference,
            output_dir / "deeplabv3_output.mp4",
            display=display
        )
        results['deeplabv3'] = {
            'fps': fps3,
            'model': 'DeepLabV3 + MobileNetV3',
            'framework': 'TorchVision',
            'success': True
        }
    except Exception as e:
        print(f"DeepLabV3 failed: {e}")
        results['deeplabv3'] = {'fps': 0, 'error': str(e), 'success': False}

    # 結果表示
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    print(f"\n{'Model':<40} {'Framework':<30} {'FPS':>10} {'Status':>10}")
    print("-" * 90)

    for name, info in results.items():
        status = "✓ OK" if info['success'] else "✗ FAIL"
        model_name = info.get('model', 'N/A')
        fps = info.get('fps', 0)
        print(f"{model_name:<40} {info.get('framework','N/A'):<30} {fps:>10.2f} {status:>10}")

    successful_results = {k: v for k, v in results.items() if v['success']}
    if successful_results:
        fastest = max(successful_results.items(), key=lambda x: x[1]['fps'])
        print(f"\nFastest: {fastest[1]['model']} ({fastest[1]['fps']:.2f} FPS)")

    results_file = output_dir / 'comparison_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {results_file}")

    markdown_file = output_dir / 'comparison_results.md'
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write("# Semantic Segmentation Comparison\n\n")
        f.write(f"**Video:** `{video_path}`\n\n")
        f.write("## Results\n\n")
        f.write("| Model | Framework | FPS | Status |\n")
        f.write("|-------|-----------|-----|--------|\n")
        for name, info in results.items():
            status = "✓" if info['success'] else "✗"
            f.write(f"| {info.get('model','N/A')} | {info.get('framework','N/A')} | {info.get('fps',0):.2f} | {status} |\n")
        if successful_results:
            f.write(f"\n**Fastest:** {fastest[1]['model']} ({fastest[1]['fps']:.2f} FPS)\n")
    print(f"Markdown report saved to: {markdown_file}")

    return results

def main():
    parser = argparse.ArgumentParser(description='Compare semantic segmentation models')
    parser.add_argument('--video', '-v', type=str,
                       default='videos/5750805-hd_1920_1080_24fps.mp4',
                       help='Input video path')
    parser.add_argument('--output-dir', '-o', type=str,
                       default='outputs',
                       help='Output directory')
    parser.add_argument('--display', '-d', action='store_true',
                       help='Display video during processing')

    args = parser.parse_args()

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
