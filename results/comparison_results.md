# Semantic Segmentation Methods Comparison

**入力動画:**
- https://drive.google.com/file/d/1Qfk6CSvBEE-0W7mw-MrW0oDiP_77vRpC/view?usp=drive_link

## 実験結果


|  Model | Framework | FPS | 
|--------|-------|-----------|
| [SegFormer-B0 ](./method1_huggingface_output.mp4) | Hugging Face Transformers  | 17.69 |
| [SegFormer-B0](./method2_segformer_b1_output.mp4) | Hugging Face Transformers | 3.36 |
| [DeepLabV3 + MobileNetV3](./method3_torchvision_output.mp4) |  TorchVision | 1.90 |

**Fastest:** method1_huggingface_segformer_b0 (17.69 FPS)
