#!/bin/bash
# 720p → 360p 変換スクリプト
# 使い方: bash convert_360p.sh input.mp4 [output.mp4]

INPUT="$1"
OUTPUT="${2:-${INPUT%.*}_360p.${INPUT##*.}}"

if [ -z "$INPUT" ]; then
    echo "使い方: bash convert_360p.sh <入力ファイル> [出力ファイル]"
    exit 1
fi

ffmpeg -i "$INPUT" -vf "scale=-2:360" -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k "$OUTPUT"

echo "完了: $OUTPUT"
