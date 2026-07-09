import argparse
import os
import random
import shutil

import cv2
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_K = 5
DEFAULT_EPOCHS = 1500
IMG_SIZE = 640
RANDOM_SEED = 42

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

DEFAULT_BASE_MODEL = "yolov8n.pt"
TRAINING_MODEL_OPTIONS = [
    {"label": "YOLOv8 (nano)", "base_model": "yolov8n.pt"},
    {"label": "YOLO11 (nano)", "base_model": "yolo11n.pt"},
    {"label": "YOLO26 (nano)", "base_model": "yolo26n.pt"},
]

OUTPUT_COLUMNS = [
    "image",
    "fold",
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
EXCEL_FILENAME = "image_analysis.xlsx"
KFOLD_WORK_DIRNAME = "kfold_work"


def _engine_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _project_root():
    return os.getcwd()


def _dataset_path():
    return os.path.join(_project_root(), "dataset")


def _kfold_work_path():
    return os.path.join(_engine_dir(), KFOLD_WORK_DIRNAME)


def _label_path_for_image(image_name):
    return image_name.rsplit(".", 1)[0] + ".txt"


def _pair_dataset_files(dataset_path):
    all_files = [
        f for f in os.listdir(dataset_path)
        if os.path.isfile(os.path.join(dataset_path, f))
    ]
    txt_files = {f for f in all_files if f.lower().endswith(".txt") and f != "classes.txt"}

    paired_files = []
    for image_file in all_files:
        if not image_file.lower().endswith(IMAGE_EXTENSIONS):
            continue
        label_file = _label_path_for_image(image_file)
        if label_file in txt_files:
            paired_files.append((image_file, label_file))

    return sorted(paired_files)


def _read_classes(dataset_path):
    classes_file = os.path.join(dataset_path, "classes.txt")
    if not os.path.exists(classes_file):
        raise FileNotFoundError("classes.txt not found in dataset folder")

    classes = []
    with open(classes_file, "r", encoding="utf-8") as f:
        for line in f:
            class_name = line.strip()
            if class_name:
                classes.append(class_name)

    if not classes:
        raise ValueError("No classes found in dataset/classes.txt")

    return classes


def _clear_output_folder():
    output_path = os.path.join(_project_root(), "output")
    if os.path.exists(output_path):
        shutil.rmtree(output_path, ignore_errors=True)
        print("Cleared output folder")
    os.makedirs(output_path, exist_ok=True)
    return output_path


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


def _save_excel(rows, output_excel_path):
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df.to_excel(output_excel_path, index=False, sheet_name="Image Analysis")


def _detect_device():
    if torch.cuda.is_available():
        print(f"Training folds on GPU: {torch.cuda.get_device_name(0)}")
        return 0
    print("WARNING: No GPU detected — folds will train on CPU (slow).")
    return "cpu"


def _build_fold_dataset(pairs, val_indices, classes, fold_idx):
    """Create an isolated train/val dataset for one fold and return (data_yaml_path, val_pairs)."""
    dataset_path = _dataset_path()
    fold_root = os.path.join(_kfold_work_path(), f"fold_{fold_idx}")

    train_images_dir = os.path.join(fold_root, "images", "train")
    val_images_dir = os.path.join(fold_root, "images", "val")
    train_labels_dir = os.path.join(fold_root, "labels", "train")
    val_labels_dir = os.path.join(fold_root, "labels", "val")

    for directory in (train_images_dir, val_images_dir, train_labels_dir, val_labels_dir):
        os.makedirs(directory, exist_ok=True)

    val_pairs = []
    for index, (image_file, label_file) in enumerate(pairs):
        if index in val_indices:
            images_dir, labels_dir = val_images_dir, val_labels_dir
            val_pairs.append(
                (
                    image_file,
                    os.path.join(dataset_path, image_file),
                    os.path.join(dataset_path, label_file),
                )
            )
        else:
            images_dir, labels_dir = train_images_dir, train_labels_dir

        shutil.copy2(
            os.path.join(dataset_path, image_file),
            os.path.join(images_dir, image_file),
        )
        shutil.copy2(
            os.path.join(dataset_path, label_file),
            os.path.join(labels_dir, label_file),
        )

    data_yaml_content = {
        "train": train_images_dir.replace(os.sep, "/"),
        "val": val_images_dir.replace(os.sep, "/"),
        "nc": len(classes),
        "names": classes,
    }
    data_yaml_path = os.path.join(fold_root, "data.yaml")
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)

    return data_yaml_path, val_pairs


def _train_fold(base_model, data_yaml_path, epochs, imgsz, device, fold_idx):
    """Train a fresh model for one fold and return the path to its best.pt."""
    work_dir = _kfold_work_path()
    run_name = f"fold_{fold_idx}"

    model = YOLO(base_model)
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        workers=8 if device != "cpu" else 0,
        project=work_dir,
        name=run_name,
        exist_ok=True,
    )

    best_weights_path = os.path.join(work_dir, run_name, "weights", "best.pt")
    if not os.path.exists(best_weights_path):
        raise FileNotFoundError(
            f"Fold {fold_idx} training finished but weights not found: {best_weights_path}"
        )
    return best_weights_path


def analyze_dataset_kfold(
    base_model=DEFAULT_BASE_MODEL,
    k=DEFAULT_K,
    epochs=DEFAULT_EPOCHS,
    imgsz=IMG_SIZE,
    conf_threshold=DEFAULT_CONF_THRESHOLD,
    iou_threshold=DEFAULT_IOU_THRESHOLD,
    output_excel_path=None,
):
    dataset_path = _dataset_path()
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    pairs = _pair_dataset_files(dataset_path)
    if len(pairs) < 2:
        raise ValueError(
            "K-Fold needs at least 2 labeled images in dataset/. "
            f"Found {len(pairs)}."
        )
    classes = _read_classes(dataset_path)

    k = max(2, min(k, len(pairs)))

    indices = list(range(len(pairs)))
    random.Random(RANDOM_SEED).shuffle(indices)
    folds = [[] for _ in range(k)]
    for position, index in enumerate(indices):
        folds[position % k].append(index)

    output_dir = _clear_output_folder()
    if output_excel_path is None:
        output_excel_path = os.path.join(output_dir, EXCEL_FILENAME)

    work_path = _kfold_work_path()
    if os.path.isdir(work_path):
        shutil.rmtree(work_path, ignore_errors=True)
    os.makedirs(work_path, exist_ok=True)

    device = _detect_device()
    print(
        f"\nK-Fold analysis: {len(pairs)} images, k={k} folds, "
        f"base_model={base_model}, epochs={epochs}"
    )
    print(f"PyTorch: {torch.__version__}")

    rows = []
    try:
        for fold_idx, val_index_list in enumerate(folds):
            if not val_index_list:
                continue

            val_set = set(val_index_list)
            print(
                f"\n===== Fold {fold_idx + 1}/{k} "
                f"({len(pairs) - len(val_set)} train, {len(val_set)} val) ====="
            )

            data_yaml_path, val_pairs = _build_fold_dataset(pairs, val_set, classes, fold_idx)
            best_weights_path = _train_fold(
                base_model, data_yaml_path, epochs, imgsz, device, fold_idx
            )

            print(f"Scoring {len(val_pairs)} held-out images for fold {fold_idx + 1}...")
            model = YOLO(best_weights_path)
            for image_name, image_path, label_path in val_pairs:
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
                        "fold": fold_idx + 1,
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

        _save_excel(rows, output_excel_path)

        print(f"\nSaved image analysis Excel: {output_excel_path}")
        if rows:
            worst = rows[0]
            print(
                f"Worst image: {worst['image']} (fold {worst['fold']}) "
                f"— error_score={worst['error_score']}, impact_rank=1"
            )

        return output_excel_path, rows
    finally:
        shutil.rmtree(work_path, ignore_errors=True)
        print("Removed temporary kfold_work folder")


def main(
    base_model=DEFAULT_BASE_MODEL,
    k=DEFAULT_K,
    epochs=DEFAULT_EPOCHS,
    conf_threshold=DEFAULT_CONF_THRESHOLD,
    iou_threshold=DEFAULT_IOU_THRESHOLD,
):
    print(f"Base model: {base_model}")
    print(f"Folds (k): {k}")
    print(f"Epochs per fold: {epochs}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU match threshold: {iou_threshold}")

    return analyze_dataset_kfold(
        base_model=base_model,
        k=k,
        epochs=epochs,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find which training images hurt accuracy via K-Fold Cross-Validation."
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"Base YOLO model to train each fold from (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"Number of folds (default: {DEFAULT_K})")
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Epochs per fold (default: {DEFAULT_EPOCHS})"
    )
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD, help="IoU match threshold")
    args = parser.parse_args()

    main(
        base_model=args.base_model,
        k=args.k,
        epochs=args.epochs,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
