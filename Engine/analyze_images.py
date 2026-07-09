"""
Single-model dataset analysis.

Workflow:
1. Select a trained YOLO model from models/<folder>/<file.pt>
2. Copy dataset/ into Engine/comparer (images/val + labels/val)
3. Run inference on every comparer image and compare against YOLO labels
4. Save annotated predictions + a 2-sheet Excel report under output/
5. Copy images with accuracy < 100 into output/low-accuracy-images/
6. Remove Engine/comparer when analysis finishes
"""

import argparse
import os
import shutil
import sys

import cv2
import pandas as pd
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configurable paths / thresholds
# ---------------------------------------------------------------------------
MODELS_DIR = "models"
DATASET_DIR = "dataset"
OUTPUT_DIR = "output"
PREDICTIONS_DIRNAME = "predictions"
LOW_ACCURACY_DIRNAME = "low-accuracy-images"
LOW_ACCURACY_THRESHOLD = 100
REPORT_FILENAME = "analysis_report.xlsx"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")

IMAGE_ANALYSIS_COLUMNS = [
    "Image Name",
    "Image Path",
    "Image Width",
    "Image Height",
    "Ground Truth Object Count",
    "Predicted Object Count",
    "Ground Truth Classes",
    "Predicted Classes",
    "Average Confidence Score",
    "Highest Confidence Score",
    "Lowest Confidence Score",
    "Missing Detection Count",
    "False Positive Count",
    "Detection Status",
    "Image Accuracy Score",
]

BOX_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def _project_root():
    return os.getcwd()


def _engine_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _models_path():
    return os.path.join(_project_root(), MODELS_DIR)


def _dataset_path():
    return os.path.join(_project_root(), DATASET_DIR)


def _comparer_path():
    return os.path.join(_engine_dir(), "comparer")


def _output_path():
    return os.path.join(_project_root(), OUTPUT_DIR)


def _predictions_path():
    return os.path.join(_output_path(), PREDICTIONS_DIRNAME)


def _low_accuracy_path():
    return os.path.join(_output_path(), LOW_ACCURACY_DIRNAME)


def _report_path():
    return os.path.join(_output_path(), REPORT_FILENAME)


def _label_path_for_image(image_name):
    return image_name.rsplit(".", 1)[0] + ".txt"


def _clear_output_folder():
    output_path = _output_path()
    if os.path.exists(output_path):
        shutil.rmtree(output_path, ignore_errors=True)
        print("Cleared output folder")
    os.makedirs(_predictions_path(), exist_ok=True)
    return output_path


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
def list_model_folders():
    models_dir = _models_path()
    if not os.path.isdir(models_dir):
        return []

    folders = []
    for name in sorted(os.listdir(models_dir)):
        folder_path = os.path.join(models_dir, name)
        if not os.path.isdir(folder_path):
            continue
        pt_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pt")]
        if pt_files:
            folders.append(name)
    return folders


def list_model_files(folder_path):
    if not os.path.isdir(folder_path):
        return []
    return sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".pt"))


def select_model_interactively():
    """Ask user to pick models/<folder>/<file.pt>."""
    folders = list_model_folders()
    if not folders:
        raise FileNotFoundError(
            f"No model folders with .pt files found in '{_models_path()}'."
        )

    print("\nSelect model:")
    print("-" * 20)
    for i, folder in enumerate(folders, 1):
        print(f"{i}. {folder}")

    while True:
        try:
            choice = input(f"\nEnter choice (1-{len(folders)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(folders):
                selected_folder = folders[choice_num - 1]
                break
            print(f"Please enter a number between 1 and {len(folders)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

    folder_path = os.path.join(_models_path(), selected_folder)
    model_files = list_model_files(folder_path)
    if not model_files:
        raise FileNotFoundError(f"No .pt files found in '{folder_path}'.")

    print(f"\nSelect version ({selected_folder}):")
    print("-" * 30)
    for i, model_file in enumerate(model_files, 1):
        print(f"{i}. {model_file}")

    while True:
        try:
            choice = input(f"\nEnter choice (1-{len(model_files)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(model_files):
                selected_model = model_files[choice_num - 1]
                break
            print(f"Please enter a number between 1 and {len(model_files)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

    model_path = os.path.join(folder_path, selected_model)
    print(f"Selected model: {selected_folder}/{selected_model}")
    return model_path


def load_class_names(model_path, model):
    """Prefer classes.txt next to the weights; fall back to model.names."""
    classes_file = os.path.join(os.path.dirname(model_path), "classes.txt")
    if os.path.exists(classes_file):
        names = []
        with open(classes_file, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    names.append(name)
        if names:
            return {i: name for i, name in enumerate(names)}

    if getattr(model, "names", None):
        return {int(k): str(v) for k, v in model.names.items()}
    return {}


# ---------------------------------------------------------------------------
# Comparer dataset preparation (same layout as Engine/compare.py)
# ---------------------------------------------------------------------------
def setup_comparer_dataset():
    """Copy dataset/ images+labels into Engine/comparer/images|labels/val."""
    dataset_path = _dataset_path()
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    comparer_path = _comparer_path()
    images_val_path = os.path.join(comparer_path, "images", "val")
    labels_val_path = os.path.join(comparer_path, "labels", "val")

    if os.path.exists(comparer_path):
        shutil.rmtree(comparer_path, ignore_errors=True)
        print("Cleared existing comparer folder")

    os.makedirs(images_val_path, exist_ok=True)
    os.makedirs(labels_val_path, exist_ok=True)

    dataset_files = [
        f for f in os.listdir(dataset_path)
        if os.path.isfile(os.path.join(dataset_path, f))
    ]
    image_files = [f for f in dataset_files if f.lower().endswith(IMAGE_EXTENSIONS)]
    text_files = [
        f for f in dataset_files
        if f.lower().endswith(".txt") and f != "classes.txt"
    ]

    print(f"Found {len(image_files)} images and {len(text_files)} labels in dataset/")

    for image_file in image_files:
        shutil.copy2(
            os.path.join(dataset_path, image_file),
            os.path.join(images_val_path, image_file),
        )
    for text_file in text_files:
        shutil.copy2(
            os.path.join(dataset_path, text_file),
            os.path.join(labels_val_path, text_file),
        )

    print(f"Comparer ready: {comparer_path}")
    return comparer_path, images_val_path, labels_val_path


# ---------------------------------------------------------------------------
# Geometry / matching helpers
# ---------------------------------------------------------------------------
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
        return boxes, False

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
    return boxes, True


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


def _class_names_joined(boxes, class_names):
    if not boxes:
        return ""
    names = []
    for box in boxes:
        class_id = box["class_id"]
        names.append(class_names.get(class_id, f"Class_{class_id}"))
    return ", ".join(names)


def detection_status(label_exists, gt_count, pred_count, fp_count, fn_count, wrong_class, mean_iou, tp_count):
    if not label_exists:
        return "Label Issue"
    if gt_count > 0 and pred_count == 0:
        return "No Detection"
    if wrong_class > 0 or (fn_count > 0 and fp_count > 0) or (tp_count > 0 and mean_iou < 0.5):
        return "Review Required"
    if fn_count == 0 and fp_count == 0:
        return "Good Detection"
    if fn_count > 0 and fp_count == 0:
        return "Missing Objects"
    if fp_count > 0 and fn_count == 0:
        return "False Detection"
    return "Review Required"


# ---------------------------------------------------------------------------
# Per-image analysis + annotated prediction save
# ---------------------------------------------------------------------------
def draw_predictions(image, pred_boxes, class_names):
    annotated = image.copy()
    for pred in pred_boxes:
        x1, y1, x2, y2 = map(int, pred["xyxy"])
        class_id = pred["class_id"]
        conf = pred["confidence"]
        color = BOX_COLORS[class_id % len(BOX_COLORS)]
        label = f"{class_names.get(class_id, f'Class_{class_id}')} {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return annotated


def analyze_single_image(model, image_path, label_path, class_names, conf_threshold, iou_threshold):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_h, img_w = image.shape[:2]
    gt_boxes, label_exists = load_yolo_labels(label_path, img_w, img_h)

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
    highest_confidence = max(confidences) if confidences else 0.0
    lowest_confidence = min(confidences) if confidences else 0.0

    status = detection_status(
        label_exists=label_exists,
        gt_count=len(gt_boxes),
        pred_count=len(pred_boxes),
        fp_count=fp_count,
        fn_count=fn_count,
        wrong_class=wrong_class,
        mean_iou=mean_iou,
        tp_count=tp_count,
    )

    row = {
        "Image Name": os.path.basename(image_path),
        "Image Path": os.path.abspath(image_path),
        "Image Width": img_w,
        "Image Height": img_h,
        "Ground Truth Object Count": len(gt_boxes),
        "Predicted Object Count": len(pred_boxes),
        "Ground Truth Classes": _class_names_joined(gt_boxes, class_names),
        "Predicted Classes": _class_names_joined(pred_boxes, class_names),
        "Average Confidence Score": round(avg_confidence, 4),
        "Highest Confidence Score": round(highest_confidence, 4),
        "Lowest Confidence Score": round(lowest_confidence, 4),
        "Missing Detection Count": fn_count,
        "False Positive Count": fp_count,
        "Detection Status": status,
        "Image Accuracy Score": round(100.0 * f1_score, 2),
    }

    annotated = draw_predictions(image, pred_boxes, class_names)
    return row, annotated


# ---------------------------------------------------------------------------
# Excel report
# ---------------------------------------------------------------------------
def write_analysis_report(rows, model_path, report_path):
    df = pd.DataFrame(rows, columns=IMAGE_ANALYSIS_COLUMNS)
    df = df.sort_values(by="Image Accuracy Score", ascending=True).reset_index(drop=True)

    total_images = len(df)
    total_objects = int(df["Ground Truth Object Count"].sum()) if total_images else 0
    total_predictions = int(df["Predicted Object Count"].sum()) if total_images else 0

    conf_series = df.loc[df["Predicted Object Count"] > 0, "Average Confidence Score"]
    average_confidence = float(conf_series.mean()) if len(conf_series) else 0.0

    failed_images = int((df["Detection Status"] != "Good Detection").sum()) if total_images else 0
    review_images = int((df["Detection Status"] == "Review Required").sum()) if total_images else 0

    worst = df.head(50)[["Image Name", "Image Accuracy Score", "Detection Status"]]

    summary_rows = [
        {"Metric": "Selected model name", "Value": os.path.basename(model_path)},
        {"Metric": "Selected model path", "Value": os.path.abspath(model_path)},
        {"Metric": "Total images analysed", "Value": total_images},
        {"Metric": "Total objects in dataset", "Value": total_objects},
        {"Metric": "Total predictions", "Value": total_predictions},
        {"Metric": "Average confidence", "Value": round(average_confidence, 4)},
        {"Metric": "Number of failed images", "Value": failed_images},
        {"Metric": "Number of images requiring review", "Value": review_images},
        {"Metric": "", "Value": ""},
        {"Metric": "Top 50 worst performing images", "Value": ""},
    ]
    for _, item in worst.iterrows():
        summary_rows.append(
            {
                "Metric": item["Image Name"],
                "Value": f"score={item['Image Accuracy Score']}, status={item['Detection Status']}",
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Image Analysis")
        summary_df.to_excel(writer, index=False, sheet_name="Summary")

    return report_path


def export_low_accuracy_images(rows, images_dir, labels_dir):
    """Copy images/labels with Image Accuracy Score below the threshold."""
    low_root = _low_accuracy_path()
    low_images_dir = os.path.join(low_root, "images")
    low_labels_dir = os.path.join(low_root, "labels")
    os.makedirs(low_images_dir, exist_ok=True)
    os.makedirs(low_labels_dir, exist_ok=True)

    exported = 0
    for row in rows:
        if row["Image Accuracy Score"] >= LOW_ACCURACY_THRESHOLD:
            continue

        image_name = row["Image Name"]
        src_image = os.path.join(images_dir, image_name)
        if not os.path.exists(src_image):
            print(f"  Warning: low-accuracy image missing, skipped: {image_name}")
            continue

        shutil.copy2(src_image, os.path.join(low_images_dir, image_name))

        label_name = _label_path_for_image(image_name)
        src_label = os.path.join(labels_dir, label_name)
        if os.path.exists(src_label):
            shutil.copy2(src_label, os.path.join(low_labels_dir, label_name))

        exported += 1

    print(
        f"Exported {exported} low-accuracy image(s) "
        f"(accuracy < {LOW_ACCURACY_THRESHOLD}) to: {low_root}"
    )
    return low_root, exported


def _remove_comparer_folder():
    comparer_path = _comparer_path()
    if os.path.isdir(comparer_path):
        shutil.rmtree(comparer_path, ignore_errors=True)
        print("Removed comparer folder")


# ---------------------------------------------------------------------------
# Main analysis driver
# ---------------------------------------------------------------------------
def run_analysis(model_path, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Selected model does not exist: {model_path}")

    print("\nRunning analysis...")
    print("-" * 30)

    _clear_output_folder()
    _, images_dir, labels_dir = setup_comparer_dataset()

    try:
        print("Loading model...")
        model = YOLO(model_path)
        class_names = load_class_names(model_path, model)

        image_files = sorted(
            f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        if not image_files:
            raise FileNotFoundError(f"No images found in comparer folder: {images_dir}")

        print(f"Processing images... ({len(image_files)} total)")
        rows = []
        predictions_dir = _predictions_path()

        for image_name in image_files:
            image_path = os.path.join(images_dir, image_name)
            label_path = os.path.join(labels_dir, _label_path_for_image(image_name))

            row, annotated = analyze_single_image(
                model,
                image_path,
                label_path,
                class_names,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            rows.append(row)

            pred_out = os.path.join(predictions_dir, image_name)
            cv2.imwrite(pred_out, annotated)
            print(
                f"  {image_name}: status={row['Detection Status']}, "
                f"accuracy={row['Image Accuracy Score']}, "
                f"FN={row['Missing Detection Count']}, FP={row['False Positive Count']}"
            )

        print("Exporting low-accuracy images...")
        low_root, _ = export_low_accuracy_images(rows, images_dir, labels_dir)

        print("Generating Excel...")
        report_path = write_analysis_report(rows, model_path, _report_path())

        print("\nCompleted:")
        print(f"  {report_path}")
        print(f"  {predictions_dir}/")
        print(f"  {low_root}/")
        return report_path, rows
    finally:
        _remove_comparer_folder()


def main(model_path=None, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD):
    if model_path is None:
        model_path = select_model_interactively()
    else:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Selected model does not exist: {model_path}")
        print(f"Using model: {model_path}")

    return run_analysis(
        model_path=model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze dataset images with a selected trained YOLO model."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to a trained .pt model (skips interactive selection)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONF_THRESHOLD,
        help=f"Confidence threshold (default: {CONF_THRESHOLD})",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=IOU_THRESHOLD,
        help=f"IoU match threshold (default: {IOU_THRESHOLD})",
    )
    args = parser.parse_args()

    main(model_path=args.model, conf_threshold=args.conf, iou_threshold=args.iou)
