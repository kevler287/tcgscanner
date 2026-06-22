import sys
import csv
import yaml
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw

def parse_args():
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py v1")
        sys.exit(1)
    return sys.argv[1]

def load_classes(dp_root):
    yaml_path = dp_root / "transform/output/data.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data["names"]

def load_ground_truth(label_path, img_w, img_h):
    masks = []
    if not label_path.exists():
        return masks
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            polygon = [(coords[i] * img_w, coords[i+1] * img_h)
                       for i in range(0, len(coords), 2)]
            mask_img = Image.new("L", (img_w, img_h), 0)
            ImageDraw.Draw(mask_img).polygon(polygon, fill=255)
            masks.append((class_id, np.array(mask_img) > 0))
    return masks

def compute_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0

def evaluate(version):
    dp_root = Path(__file__).parent
    model_path = dp_root / f"transform/{version}/{version}.pt"
    test_img_dir = dp_root / "transform/output/images/test"
    test_lbl_dir = dp_root / "transform/output/labels/test"
    results_dir = dp_root / f"transform/{version}/results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    classes = load_classes(dp_root)
    model = YOLO(str(model_path))

    image_paths = sorted(test_img_dir.glob("*.[jp][pn]g"))
    if not image_paths:
        print(f"No images found in {test_img_dir}")
        sys.exit(1)

    rows = []
    class_ious = {cls: [] for cls in classes.values()}

    print(f"Evaluating {len(image_paths)} images...")

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        label_path = test_lbl_dir / (img_path.stem + ".txt")
        gt_masks = load_ground_truth(label_path, img_w, img_h)

        result = model(img_path, verbose=False)[0]

        if result.masks is None or len(result.masks) == 0:
            rows.append({
                "image": img_path.name,
                "class": "none",
                "confidence": 0.0,
                "iou": 0.0,
                "gt_mask_coverage_%": 0.0,
                "pred_mask_coverage_%": 0.0,
                "status": "no_detection"
            })
            continue

        pred_classes = result.boxes.cls.cpu().numpy().astype(int)
        pred_confs = result.boxes.conf.cpu().numpy()
        pred_masks_raw = result.masks.data.cpu().numpy()

        for i, (cls_id, conf, pred_mask_small) in enumerate(
                zip(pred_classes, pred_confs, pred_masks_raw)):

            pred_mask = np.array(
                Image.fromarray((pred_mask_small * 255).astype(np.uint8)).resize(
                    (img_w, img_h), Image.NEAREST)) > 127

            best_iou = 0.0
            gt_coverage = 0.0
            for gt_cls, gt_mask in gt_masks:
                if gt_cls == cls_id:
                    iou = compute_iou(pred_mask, gt_mask)
                    if iou > best_iou:
                        best_iou = iou
                        gt_coverage = gt_mask.sum() / (img_w * img_h) * 100

            pred_coverage = pred_mask.sum() / (img_w * img_h) * 100
            class_name = classes.get(cls_id, str(cls_id))
            class_ious[class_name].append(best_iou)

            rows.append({
                "image": img_path.name,
                "class": class_name,
                "confidence": round(float(conf), 4),
                "iou": round(best_iou, 4),
                "gt_mask_coverage_%": round(gt_coverage, 2),
                "pred_mask_coverage_%": round(pred_coverage, 2),
                "status": "ok"
            })

    csv_path = results_dir / "results.csv"
    fieldnames = ["image", "class", "confidence", "iou",
                  "gt_mask_coverage_%", "pred_mask_coverage_%", "status"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = results_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Evaluation summary — {version}\n")
        f.write(f"{'='*40}\n")
        f.write(f"Total images: {len(image_paths)}\n")
        f.write(f"Total detections: {len(rows)}\n\n")
        f.write("IoU per class:\n")
        for cls_name, ious in class_ious.items():
            if ious:
                f.write(f"  {cls_name}: mean={np.mean(ious):.4f}  "
                        f"min={np.min(ious):.4f}  max={np.max(ious):.4f}  "
                        f"n={len(ious)}\n")
            else:
                f.write(f"  {cls_name}: no detections\n")

    print(f"\nDone. Results saved to {results_dir}")
    print(f"  → {csv_path}")
    print(f"  → {summary_path}")

    print("\nSummary:")
    for cls_name, ious in class_ious.items():
        if ious:
            print(f"  {cls_name}: mean IoU = {np.mean(ious):.4f} "
                  f"(n={len(ious)})")

if __name__ == "__main__":
    version = parse_args()
    evaluate(version)