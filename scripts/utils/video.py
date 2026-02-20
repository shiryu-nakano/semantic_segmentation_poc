import time
import cv2
import numpy as np
from .colormap import visualize_segmentation


def process_video(video_path, model_inference_fn, output_path, display=True):
    """動画処理の共通関数"""
    cap = cv2.VideoCapture(str(video_path))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    fps_list = []

    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        seg_mask = model_inference_fn(frame)
        inference_time = time.time() - start
        fps_list.append(1.0 / inference_time)

        result = visualize_segmentation(frame, seg_mask)

        cv2.putText(result, f"FPS: {fps_list[-1]:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(result)

        if display:
            cv2.imshow('Segmentation', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames, Avg FPS: {np.mean(fps_list):.2f}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nAverage FPS: {np.mean(fps_list):.2f}")
    print(f"Output saved to: {output_path}")

    return np.mean(fps_list)
