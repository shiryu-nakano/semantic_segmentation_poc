# Semantic Segmentation PoC

リアルタイムセマンティックセグメンテーションのPoC検証

## Setup
```bash
pyenv local 3.10.13
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Methods
- Method 1: Hugging Face SegFormer
- Method 2: MMSegmentation
- Method 3: NVIDIA TAO

## Cityscapes Class Colors

| Class | Color (RGB) | Preview |
|-------|-------------|---------|
| road | `[128, 64, 128]` | ![#804080](https://via.placeholder.com/15/804080/000000?text=+) |
| sidewalk | `[244, 35, 232]` | ![#F423E8](https://via.placeholder.com/15/F423E8/000000?text=+) |
| building | `[70, 70, 70]` | ![#464646](https://via.placeholder.com/15/464646/000000?text=+) |
| wall | `[102, 102, 156]` | ![#66669C](https://via.placeholder.com/15/66669C/000000?text=+) |
| fence | `[190, 153, 153]` | ![#BE9999](https://via.placeholder.com/15/BE9999/000000?text=+) |
| pole | `[153, 153, 153]` | ![#999999](https://via.placeholder.com/15/999999/000000?text=+) |
| traffic light | `[250, 170, 30]` | ![#FAAA1E](https://via.placeholder.com/15/FAAA1E/000000?text=+) |
| traffic sign | `[220, 220, 0]` | ![#DCDC00](https://via.placeholder.com/15/DCDC00/000000?text=+) |
| vegetation | `[107, 142, 35]` | ![#6B8E23](https://via.placeholder.com/15/6B8E23/000000?text=+) |
| terrain | `[152, 251, 152]` | ![#98FB98](https://via.placeholder.com/15/98FB98/000000?text=+) |
| sky | `[70, 130, 180]` | ![#4682B4](https://via.placeholder.com/15/4682B4/000000?text=+) |
| person | `[220, 20, 60]` | ![#DC143C](https://via.placeholder.com/15/DC143C/000000?text=+) |
| rider | `[255, 0, 0]` | ![#FF0000](https://via.placeholder.com/15/FF0000/000000?text=+) |
| car | `[0, 0, 142]` | ![#00008E](https://via.placeholder.com/15/00008E/000000?text=+) |
| truck | `[0, 0, 70]` | ![#000046](https://via.placeholder.com/15/000046/000000?text=+) |
| bus | `[0, 60, 100]` | ![#003C64](https://via.placeholder.com/15/003C64/000000?text=+) |
| train | `[0, 80, 100]` | ![#005064](https://via.placeholder.com/15/005064/000000?text=+) |
| motorcycle | `[0, 0, 230]` | ![#0000E6](https://via.placeholder.com/15/0000E6/000000?text=+) |
| bicycle | `[119, 11, 32]` | ![#770B20](https://via.placeholder.com/15/770B20/000000?text=+) |

## Video Source
https://www.pexels.com/search/videos/driving/
