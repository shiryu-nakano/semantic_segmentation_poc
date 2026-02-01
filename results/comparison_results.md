# Semantic Segmentation Methods Comparison

**Video:** `videos/5750805-hd_1920_1080_24fps.mp4`

## Results

| Method | Model | Framework | FPS | Status |
|--------|-------|-----------|-----|--------|
| method1_huggingface_segformer_b0 | SegFormer-B0 | Hugging Face Transformers | 17.69 | ✓ |
| method2_segformer_b1 | SegFormer-B1 | Hugging Face Transformers | 3.36 | ✓ |
| method3_torchvision_deeplabv3 | DeepLabV3 + MobileNetV3 | TorchVision | 1.90 | ✓ |

**Fastest:** method1_huggingface_segformer_b0 (17.69 FPS)
