#!/bin/bash
# 動画を1分刻みで分割するスクリプト
# 使い方: bash split_video.sh input.mp4

INPUT="$1"

if [ -z "$INPUT" ]; then
    echo "使い方: bash split_video.sh <入力ファイル>"
    exit 1
fi

# 入力ファイル名（拡張子なし）と拡張子を取得
BASENAME=$(basename "$INPUT")
NAME="${BASENAME%.*}"
EXT="${BASENAME##*.}"

# 出力ディレクトリを作成
OUTDIR="$NAME"
mkdir -p "$OUTDIR"

# 動画の総秒数を取得
DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$INPUT" | cut -d. -f1)

if [ -z "$DURATION" ] || [ "$DURATION" -eq 0 ]; then
    echo "エラー: 動画の長さを取得できませんでした"
    exit 1
fi

echo "動画の長さ: ${DURATION}秒"

SEGMENT=60  # 1分 = 60秒
N=1
START=0

while [ "$START" -lt "$DURATION" ]; do
    OUTFILE="${OUTDIR}/${NAME}-${N}.mp4"
    echo "分割中: ${OUTFILE} (${START}秒〜)"

    ffmpeg -y -i "$INPUT" -ss "$START" -t "$SEGMENT" -c copy -avoid_negative_ts make_zero "$OUTFILE" 2>/dev/null

    START=$((START + SEGMENT))
    N=$((N + 1))
done

echo "完了: $((N - 1))個のファイルを ${OUTDIR}/ に保存しました"
