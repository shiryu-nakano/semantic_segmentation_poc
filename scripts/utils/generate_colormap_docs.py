# generate_colormap_docs.py (修正版)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from utils.colormap import create_color_map

def rgb_to_hex(rgb):
    """RGB to HEX変換（大文字）"""
    return '#{:02X}{:02X}{:02X}'.format(rgb[0], rgb[1], rgb[2])

def get_class_labels(lang='en'):
    """クラスラベルを言語別に取得"""
    labels_en = [
        'road', 'sidewalk', 'building', 'wall', 'fence',
        'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
        'sky', 'person', 'rider', 'car', 'truck',
        'bus', 'train', 'motorcycle', 'bicycle'
    ]
    
    labels_ja = [
        '道路', '歩道', '建物', '壁', 'フェンス',
        'ポール', '信号機', '交通標識', '植生', '地形',
        '空', '人', 'ライダー', '車', 'トラック',
        'バス', '電車', 'バイク', '自転車'
    ]
    
    return labels_ja if lang == 'ja' else labels_en

def generate_markdown_table(lang='en', output_file=None):
    """Markdownテーブルを生成"""
    colors = create_color_map(19)
    labels = get_class_labels(lang)
    
    # タイトルと見出しを言語別に設定
    if lang == 'ja':
        title = "## Cityscapes クラスカラー\n\n"
        headers = "| クラス | RGB | Hex | カラー |\n"
    else:
        title = "## Cityscapes Class Colors\n\n"
        headers = "| Class | RGB | Hex | Color |\n"
    
    markdown = title + headers
    markdown += "|-------|-----|-----|----------|\n"
    
    for label, rgb in zip(labels, colors):
        hex_color = rgb_to_hex(rgb)
        # シンプルな色表示（バッジスタイル）
        color_badge = f'![](https://img.shields.io/badge/color-{hex_color[1:]}-{hex_color[1:]})'
        markdown += f"| {label} | `{rgb.tolist()}` | `{hex_color}` | {color_badge} |\n"
    
    # ファイル出力
    if output_file is None:
        output_file = f'COLORMAP_{lang.upper()}.md'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    print(f"Colormap documentation saved to {output_file}")
    print("\n" + markdown)
    
    return markdown

def generate_html_table(lang='en', output_file=None):
    """HTMLテーブルを生成"""
    colors = create_color_map(19)
    labels = get_class_labels(lang)
    
    # タイトルと見出しを言語別に設定
    if lang == 'ja':
        title = "Cityscapes クラスカラー"
        col_headers = ['クラス', 'RGB', 'Hex', 'カラー']
    else:
        title = "Cityscapes Class Colors"
        col_headers = ['Class', 'RGB', 'Hex', 'Color']
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 900px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 30px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 15px;
            text-align: left;
            border: 1px solid #ddd;
        }}
        th {{
            background-color: #2c3e50;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #e8f4f8;
        }}
        .color-box {{
            width: 80px;
            height: 30px;
            border: 2px solid #333;
            display: inline-block;
            border-radius: 4px;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <table>
        <thead>
            <tr>
                <th>{col_headers[0]}</th>
                <th>{col_headers[1]}</th>
                <th>{col_headers[2]}</th>
                <th>{col_headers[3]}</th>
            </tr>
        </thead>
        <tbody>
"""
    
    for label, rgb in zip(labels, colors):
        hex_color = rgb_to_hex(rgb)
        html += f"""            <tr>
                <td><strong>{label}</strong></td>
                <td><code>{rgb.tolist()}</code></td>
                <td><code>{hex_color}</code></td>
                <td><div class="color-box" style="background-color: {hex_color};"></div></td>
            </tr>
"""
    
    html += """        </tbody>
    </table>
</body>
</html>
"""
    
    # ファイル出力
    if output_file is None:
        output_file = f'colormap_{lang}.html'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"HTML colormap saved to {output_file}")
    
    return html

def main():
    parser = argparse.ArgumentParser(description='Cityscapesカラーマップドキュメント生成')
    parser.add_argument('--lang', '-l', 
                       choices=['en', 'ja'], 
                       default='en',
                       help='言語選択 (en: English, ja: 日本語)')
    parser.add_argument('--format', '-f',
                       choices=['markdown', 'html', 'both'],
                       default='both',
                       help='出力フォーマット')
    parser.add_argument('--output', '-o',
                       help='出力ファイル名（指定しない場合は自動生成）')
    
    args = parser.parse_args()
    
    if args.format in ['markdown', 'both']:
        generate_markdown_table(args.lang, args.output if args.format == 'markdown' else None)
    
    if args.format in ['html', 'both']:
        generate_html_table(args.lang, args.output if args.format == 'html' else None)

if __name__ == "__main__":
    main()
