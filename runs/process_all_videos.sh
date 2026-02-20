#!/bin/bash

# 使用方法 (プロジェクトルートから実行):
# ./runs/process_all_videos.sh tsukuba2025/tsukuba2025-360

# スクリプトの場所からプロジェクトルートへ移動
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

if [ $# -eq 0 ]; then
    echo "Usage: $0 <video_directory>"
    echo "Example: $0 tsukuba2025/tsukuba2025-360"
    exit 1
fi

VIDEO_DIR="$1"
VIDEOS_BASE="videos"

# 指定されたディレクトリが存在するか確認
if [ ! -d "${VIDEOS_BASE}/${VIDEO_DIR}" ]; then
    echo "Error: Directory ${VIDEOS_BASE}/${VIDEO_DIR} does not exist"
    exit 1
fi

# ディレクトリ内のすべての.mp4ファイルを処理
echo "Processing all videos in ${VIDEO_DIR}..."
echo "========================================"

count=0
for video in "${VIDEOS_BASE}/${VIDEO_DIR}"/*.mp4; do
    if [ -f "$video" ]; then
        # videos/を取り除いて相対パスを取得
        relative_path="${video#${VIDEOS_BASE}/}"

        echo ""
        echo "[$((count+1))] Processing: ${relative_path}"
        echo "----------------------------------------"

        python scripts/method1_huggingface.py "${relative_path}"

        if [ $? -eq 0 ]; then
            echo "✓ Successfully processed: ${relative_path}"
        else
            echo "✗ Failed to process: ${relative_path}"
        fi

        count=$((count+1))
    fi
done

echo ""
echo "========================================"
echo "Total videos processed: ${count}"
