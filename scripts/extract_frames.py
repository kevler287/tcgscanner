"""
Extrahiert jeden x-ten Frame aus einem Video und speichert ihn als Bild.

Nutzung:
    python extract_frames.py --video pfad/zum/video.mp4 --output frames/ --step 10
"""

import argparse
import os
import cv2


def extract_frames(video_path: str, output_dir: str, step: int, prefix: str = "frame"):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video nicht gefunden: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Video konnte nicht geöffnet werden: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {video_path}")
    print(f"Frames gesamt: {total_frames}, FPS: {fps:.2f}")
    print(f"Speichere jeden {step}. Frame nach '{output_dir}'")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            filename = f"{prefix}_{frame_idx:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"Fertig. {saved_count} Frames gespeichert.")


def main():
    parser = argparse.ArgumentParser(description="Extrahiert jeden x-ten Frame aus einem Video.")
    parser.add_argument("--video", required=True, help="Pfad zur Videodatei")
    parser.add_argument("--output", default="frames", help="Ausgabeverzeichnis (default: frames/)")
    parser.add_argument("--step", type=int, default=10, help="Jeden x-ten Frame speichern (default: 10)")
    parser.add_argument("--prefix", default="frame", help="Dateiname-Präfix (default: frame)")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.step, args.prefix)


if __name__ == "__main__":
    main()