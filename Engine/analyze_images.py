import argparse
import csv
import os

import cv2
import yaml
from ultralytics import YOLO

from Engine.train import IMAGE_EXTENSIONS, prepare_comparer_dataset, write_data_yaml, read_classes

DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.5
CSV_COLUMNS = [
    "image",
    "split",
    "gt_boxes",
    "pred_boxes",
    "true_positives",
    "false_positives",
    "false_negatives",
    "wrong_class",
    "mean_iou",
    "min_iou",
    "precision",
    "recall",
    "f1_score",
    "avg_confidence",
    "error_score",
    "impact_rank",
]


def _engine_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _project_root():
    return os.getcwd()


def _label_path_for_image(image_name):
    return image_name.rsplit(".", 1)[0] + ".txt"


def _xywhn_to_xyxy(cx, cy, width, height, img_w, img_h):
    x1 = (cx - width / 2) * img_w
    y1 = (cy - height / 2) * img_h
    x2 = (cx + width / 2) * img_w
    y2 = (cy + height / 2) * img_h
    return [x1, y1, x2, y2]


def _box_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection_w = max(0.0, x2 - x1)
    intersection_h = max(0.0, y2 - y1)
    intersection = intersection_w * intersection_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def load_yolo_labels(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            cx, cy, width, height = map(float, parts[1:5])
            boxes.append(
                {
                    "class_id": class_id,
                    "xyxy": _xywhn_to_xyxy(cx, cy, width, height, img_w, img_h),
                }
            )
    return boxes


def match_predictions(gt_boxes, pred_boxes, iou_threshold):
    """Greedy match: highest-confidence predictions first."""
    sorted_preds = sorted(pred_boxes, key=lambda item: item["confidence"], reverse=True)
    matched_gt = set()
    matched_pairs = []
    false_positives = []
    wrong_class = 0

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt in enumerate(gt_boxes):
            if gt_index in matched_gt:
                continue
            iou = _box_iou(pred["xyxy"], gt["xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_gt_index is None or best_iou < iou_threshold:
            false_positives.append(pred)
            continue

        gt = gt_boxes[best_gt_index]
        if pred["class_id"] != gt["class_id"]:
            wrong_class += 1
            false_positives.append(pred)
            continue

        matched_gt.add(best_gt_index)
        matched_pairs.append(
            {
                "iou": best_iou,
                "pred": pred,
                "gt": gt,
                "class_match": True,
            }
        )

    false_negatives = [gt for index, gt in enumerate(gt_boxes) if index not in matched_gt]
    true_positives = [pair for pair in matched_pairs if pair["class_match"]]

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "wrong_class": wrong_class,
    }


def _safe_ratio(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def analyze_image(model, image_path, label_path, conf_threshold, iou_threshold):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_h, img_w = image.shape[:2]
    gt_boxes = load_yolo_labels(label_path, img_w, img_h)

    results = model.predict(image_path, conf=conf_threshold, verbose=False)
    pred_boxes = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            pred_boxes.append(
                {
                    "class_id": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "xyxy": box.xyxy[0].tolist(),
                }
            )

    match = match_predictions(gt_boxes, pred_boxes, iou_threshold)
    tp_count = len(match["true_positives"])
    fp_count = len(match["false_positives"])
    fn_count = len(match["false_negatives"])
    wrong_class = match["wrong_class"]

    ious = [pair["iou"] for pair in match["true_positives"]]
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    min_iou = min(ious) if ious else 0.0

    precision = _safe_ratio(tp_count, tp_count + fp_count)
    recall = _safe_ratio(tp_count, tp_count + fn_count)
    if precision is None:
        precision = 0.0 if fp_count else 1.0
    if recall is None:
        recall = 0.0 if fn_count else 1.0
    f1_score = _safe_ratio(2 * precision * recall, precision + recall)
    if f1_score is None:
        f1_score = 0.0

    confidences = [pred["confidence"] for pred in pred_boxes]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    localization_penalty = tp_count * (1.0 - mean_iou) if tp_count else 0.0
    error_score = fp_count + fn_count + localization_penalty

    return {
        "gt_boxes": len(gt_boxes),
        "pred_boxes": len(pred_boxes),
        "true_positives": tp_count,
        "false_positives": fp_count,
        "false_negatives": fn_count,
        "wrong_class": wrong_class,
        "mean_iou": round(mean_iou, 4),
        "min_iou": round(min_iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "avg_confidence": round(avg_confidence, 4),
        "error_score": round(error_score, 4),
    }


def _resolve_split_paths(data_yaml_path, split):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    images_rel = data.get(split)
    if not images_rel:
        raise ValueError(f"Split '{split}' not found in {data_yaml_path}")

    engine_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    images_rel = images_rel.replace("\\", "/")
    labels_rel = images_rel.replace(f"images/{split}", f"labels/{split}")
    images_dir = os.path.normpath(os.path.join(engine_dir, images_rel.replace("/", os.sep)))
    labels_dir = os.path.normpath(os.path.join(engine_dir, labels_rel.replace("/", os.sep)))

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    return images_dir, labels_dir


def _ensure_comparer_dataset():
    comparer_path = os.path.join(_engine_dir(), "comparer")
    data_yaml_path = os.path.join(_engine_dir(), "data.yaml")

    if os.path.isdir(comparer_path) and os.path.exists(data_yaml_path):
        return data_yaml_path

    dataset_path = os.path.join(_project_root(), "dataset")
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            "No comparer dataset found. Train a model first or place paired files in dataset/."
        )

    classes, _ = read_classes(dataset_path)
    prepare_comparer_dataset(dataset_path)
    return write_data_yaml(classes)


def analyze_dataset(
    model_path,
    output_csv_path=None,
    splits=None,
    conf_threshold=DEFAULT_CONF_THRESHOLD,
    iou_threshold=DEFAULT_IOU_THRESHOLD,
):
    if splits is None:
        splits = ("val",)

    if output_csv_path is None:
        output_dir = os.path.join(_project_root(), "output")
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "image_analysis.csv")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    data_yaml_path = _ensure_comparer_dataset()
    model = YOLO(model_path)

    rows = []
    for split in splits:
        images_dir, labels_dir = _resolve_split_paths(data_yaml_path, split)
        image_files = sorted(
            f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTENSIONS)
        )

        print(f"\nAnalyzing {len(image_files)} {split} images...")
        for image_name in image_files:
            image_path = os.path.join(images_dir, image_name)
            label_path = os.path.join(labels_dir, _label_path_for_image(image_name))
            metrics = analyze_image(
                model,
                image_path,
                label_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            rows.append(
                {
                    "image": image_name,
                    "split": split,
                    **metrics,
                }
            )
            print(
                f"  {image_name}: error_score={metrics['error_score']:.2f}, "
                f"FP={metrics['false_positives']}, FN={metrics['false_negatives']}, "
                f"mean_iou={metrics['mean_iou']:.2f}"
            )

    rows.sort(key=lambda row: row["error_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["impact_rank"] = rank

    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved image analysis CSV: {output_csv_path}")
    if rows:
        worst = rows[0]
        print(
            f"Worst image: {worst['image']} ({worst['split']}) "
            f"— error_score={worst['error_score']}, impact_rank=1"
        )

    return output_csv_path, rows


def _default_model_path():
    candidates = [
        os.path.join(_project_root(), "output", "best.pt"),
        os.path.join(_project_root(), "output", "training", "weights", "best.pt"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def main(model_path=None, splits=None, conf_threshold=DEFAULT_CONF_THRESHOLD, iou_threshold=DEFAULT_IOU_THRESHOLD):
    if model_path is None:
        model_path = _default_model_path()
    if model_path is None:
        raise FileNotFoundError(
            "No model path provided and output/best.pt was not found. "
            "Train a model or pass a .pt file path."
        )

    print(f"Using model: {model_path}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU match threshold: {iou_threshold}")

    return analyze_dataset(
        model_path=model_path,
        splits=splits or ("val",),
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze per-image impact on YOLO model accuracy.")
    parser.add_argument("--model", help="Path to trained .pt weights (default: output/best.pt)")
    parser.add_argument(
        "--split",
        choices=("val", "train", "both"),
        default="val",
        help="Dataset split to analyze (default: val)",
    )
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD, help="IoU match threshold")
    args = parser.parse_args()

    split_arg = ("val", "train") if args.split == "both" else (args.split,)
    main(
        model_path=args.model,
        splits=split_arg,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
