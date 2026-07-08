from ultralytics import YOLO
import pandas as pd
import os
import re
import shutil
import yaml
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

VAL_PLOT_FILES = [
    "confusion_matrix_normalized.png",
    "confusion_matrix.png",
    "F1_curve.png",
    "P_curve.png",
    "PR_curve.png",
    "R_curve.png",
]

TARGET_CELL_WIDTH = 520
CELL_PADDING = 6
LABEL_HEIGHT = 28

METRIC_COLUMNS = ["Precision", "Recall", "mAP50", "mAP50-95"]
METRIC_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def _model_sort_key(name):
    match = re.search(r"\d+", name)
    return int(match.group()) if match else name


def _crop_whitespace(img, threshold=250):
    """Crop borders around non-white plot content."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray < threshold
    if not mask.any():
        return img
    coords = np.column_stack(np.where(mask))
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return img[y0:y1 + 1, x0:x1 + 1]


def _make_labeled_cell(model_name, img):
    """Crop, scale, and add a compact title bar — no extra vertical padding."""
    cropped = _crop_whitespace(img)
    h, w = cropped.shape[:2]
    scale = TARGET_CELL_WIDTH / w if w > TARGET_CELL_WIDTH else 1.0
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cell = np.full((new_h + LABEL_HEIGHT, new_w, 3), 255, dtype=np.uint8)
    cell[LABEL_HEIGHT:, :] = resized

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(model_name, font, 0.55, 2)[0]
    text_x = max(0, (new_w - text_size[0]) // 2)
    text_y = (LABEL_HEIGHT + text_size[1]) // 2
    cv2.putText(cell, model_name, (text_x, text_y), font, 0.55, (0, 0, 0), 2)
    return cell


def combine_val_plots_grid(model_run_dirs, plot_filename, output_path):
    """Stitch the same YOLO val plot from each model into a labeled grid image."""
    cells = []
    for model_name, run_dir in model_run_dirs.items():
        img_path = os.path.join(run_dir, plot_filename)
        if not os.path.exists(img_path):
            print(f"  Warning: missing {plot_filename} for {model_name}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Warning: could not read {img_path}")
            continue
        cells.append(_make_labeled_cell(model_name, img))

    if not cells:
        print(f"  Warning: no images found for {plot_filename}, skipping grid.")
        return False

    n = len(cells)
    cols = min(n, 3) if n > 2 else n
    rows = math.ceil(n / cols)

    row_heights = []
    col_widths = [0] * cols
    for idx, cell in enumerate(cells):
        row, col = divmod(idx, cols)
        cell_h, cell_w = cell.shape[:2]
        if row >= len(row_heights):
            row_heights.append(cell_h)
        else:
            row_heights[row] = max(row_heights[row], cell_h)
        col_widths[col] = max(col_widths[col], cell_w)

    canvas_h = sum(row_heights) + CELL_PADDING * max(0, rows - 1)
    canvas_w = sum(col_widths) + CELL_PADDING * max(0, cols - 1)
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    row_y = 0
    for row in range(rows):
        col_x = 0
        row_h = row_heights[row]
        for col in range(cols):
            idx = row * cols + col
            if idx >= n:
                break
            cell = cells[idx]
            cell_h, cell_w = cell.shape[:2]
            y_offset = (row_h - cell_h) // 2
            canvas[row_y + y_offset:row_y + y_offset + cell_h, col_x:col_x + cell_w] = cell
            col_x += col_widths[col] + CELL_PADDING
        row_y += row_h + CELL_PADDING

    cv2.imwrite(output_path, canvas)
    return True


def generate_metrics_line_chart(df, output_path):
    """Line chart: models on X-axis, four metric series with labeled points."""
    plot_df = df.dropna(how="all")
    if plot_df.empty:
        print("  Warning: no valid metrics for line chart, skipping.")
        return False

    plot_df = plot_df.sort_index(key=lambda names: [_model_sort_key(n) for n in names])
    models = plot_df.index.tolist()
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 6))
    for metric, color in zip(METRIC_COLUMNS, METRIC_COLORS):
        if metric not in plot_df.columns:
            continue
        values = plot_df[metric].astype(float).values
        ax.plot(x, values, marker="o", linewidth=2, label=metric, color=color)
        for xi, yi in zip(x, values):
            if np.isfinite(yi):
                ax.annotate(
                    f"{yi:.2f}",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    color=color,
                )

    ax.set_title("Model Metrics Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def clear_runs_folder():
    """Remove the runs folder before comparison so val outputs start fresh."""
    runs_path = os.path.join(os.getcwd(), "runs")
    if os.path.exists(runs_path):
        shutil.rmtree(runs_path)
        print("Cleared runs folder")
    else:
        print("Runs folder does not exist, will be created by YOLO")


def clear_output_folder():
    """Remove the output folder before comparison so results start fresh."""
    output_path = os.path.join(os.getcwd(), "output")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        print("Cleared output folder")
    else:
        print("Output folder does not exist, will be created")


def clear_comparer_folder():
    """Remove the temporary comparer dataset folder after comparison."""
    comparer_path = os.path.join(os.path.dirname(__file__), "comparer")
    if os.path.exists(comparer_path):
        shutil.rmtree(comparer_path)
        print("Cleared comparer folder")


def create_data_yaml(folder_path):
    """Create data.yaml file based on the model folder and classes.txt"""
    # Read classes from classes.txt
    classes_file = os.path.join(folder_path, "classes.txt")
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"classes.txt not found in {folder_path}")
    
    classes = []
    with open(classes_file, 'r') as f:
        for line in f:
            class_name = line.strip()
            if class_name:
                classes.append(class_name)
    
    if not classes:
        raise ValueError(f"No classes found in {classes_file}")
    
    # Create data.yaml content
    data_yaml_content = {
        'train': 'comparer/images/train',
        'val': 'comparer/images/val',
        'nc': len(classes),
        'names': classes
    }
    
    # Save data.yaml in Engine folder
    data_yaml_path = os.path.join(os.path.dirname(__file__), "data.yaml")
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)
    
    print(f"Created data.yaml with {len(classes)} classes: {classes}")
    return data_yaml_path, classes

def setup_comparer_dataset():
    """Setup comparer dataset by moving files from dataset folder to comparer structure"""
    dataset_path = os.path.join(os.getcwd(), 'dataset')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
    
    # Create comparer folder structure
    comparer_path = os.path.join(os.path.dirname(__file__), "comparer")
    images_val_path = os.path.join(comparer_path, "images", "val")
    labels_val_path = os.path.join(comparer_path, "labels", "val")
    
    # Clear existing comparer folder if it exists
    if os.path.exists(comparer_path):
        shutil.rmtree(comparer_path)
        print("Cleared existing comparer folder")
    
    # Create directories
    os.makedirs(images_val_path, exist_ok=True)
    os.makedirs(labels_val_path, exist_ok=True)
    
    # Get all files from dataset folder
    dataset_files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
    
    # Separate image and text files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    image_files = [f for f in dataset_files if f.lower().endswith(image_extensions)]
    text_files = [f for f in dataset_files if f.lower().endswith('.txt')]
    
    print(f"Found {len(image_files)} image files and {len(text_files)} text files in dataset")
    
    # Copy image files to comparer/images/val
    for image_file in image_files:
        src_path = os.path.join(dataset_path, image_file)
        dst_path = os.path.join(images_val_path, image_file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied image: {image_file}")
    
    # Copy text files to comparer/labels/val
    for text_file in text_files:
        src_path = os.path.join(dataset_path, text_file)
        dst_path = os.path.join(labels_val_path, text_file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied label: {text_file}")
    
    print(f"Dataset setup complete. Files copied to: {comparer_path}")
    return comparer_path

def compare_models(folder_path):
    """Compare all models in the specified folder"""
    clear_runs_folder()
    clear_output_folder()

    # Setup comparer dataset first
    print("Setting up comparer dataset...")
    comparer_path = setup_comparer_dataset()

    try:
        # Create data.yaml file
        data_yaml_path, dataset_classes = create_data_yaml(folder_path)
        data_nc = len(dataset_classes)
        val_project = os.path.join(os.getcwd(), "runs", "detect")

        # Get all .pt files in the folder
        model_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
        if not model_files:
            print(f"No .pt model files found in {folder_path}")
            return

        # Create model paths dictionary
        model_paths = {}
        for model_file in sorted(model_files):
            model_name = os.path.splitext(model_file)[0]
            model_paths[model_name] = os.path.join(folder_path, model_file)

        print(f"Found {len(model_paths)} models to compare:")
        for name, path in model_paths.items():
            print(f"  - {name}: {os.path.basename(path)}")

        # Evaluation metrics we care about
        metrics_to_extract = {
            "metrics/precision(B)": "Precision",
            "metrics/recall(B)": "Recall",
            "metrics/mAP50(B)": "mAP50",
            "metrics/mAP50-95(B)": "mAP50-95"
        }

        # Evaluate and collect results
        results_summary = {}
        model_run_dirs = {}

        for model_name, model_path in model_paths.items():
            try:
                print(f"\nEvaluating {model_name}...")
                model = YOLO(model_path)
                model_nc = model.model.nc
                if model_nc != data_nc:
                    model_classes = list(model.names.values())
                    print(
                        f"⚠️ Skipping {model_name}: model has {model_nc} classes "
                        f"{model_classes}, but dataset has {data_nc} classes "
                        f"{dataset_classes}"
                    )
                    results_summary[model_name] = {v: None for v in metrics_to_extract.values()}
                    continue

                run_name = os.path.splitext(os.path.basename(model_path))[0]
                val_save_dir = os.path.join(val_project, run_name)
                results = model.val(
                    data=data_yaml_path,
                    split="val",
                    verbose=False,
                    project=val_project,
                    name=run_name,
                )
                metrics = results.results_dict

                # Extract only the metrics we care about, fallback to None if missing
                filtered_metrics = {
                    metrics_to_extract[k]: metrics.get(k, None) for k in metrics_to_extract
                }

                results_summary[model_name] = filtered_metrics
                model_run_dirs[model_name] = val_save_dir

            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")
                results_summary[model_name] = {v: None for v in metrics_to_extract.values()}

        # Create a comparison DataFrame
        df = pd.DataFrame(results_summary).T
        print("\n🔍 Model Comparison:")
        print(df.round(4))

        # Save results to output folder
        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)

        # Save comparison results
        results_file = os.path.join(output_dir, "model_comparison.csv")
        df.to_csv(results_file)
        print(f"\n📊 Comparison results saved to: {results_file}")

        metrics_chart = os.path.join(output_dir, "model_metrics_comparison.png")
        if generate_metrics_line_chart(df, metrics_chart):
            print(f"📈 Metrics line chart saved to: {metrics_chart}")

        if model_run_dirs:
            print("\n📈 Generating comparison grid images...")
            for plot_file in VAL_PLOT_FILES:
                out_name = plot_file.replace(".png", "_comparison.png")
                out_path = os.path.join(output_dir, out_name)
                if combine_val_plots_grid(model_run_dirs, plot_file, out_path):
                    print(f"  Comparison grid saved: {out_path}")
        else:
            print("\n⚠️ No successful model runs — skipping comparison grid images.")
    finally:
        clear_comparer_folder()

# For backward compatibility, keep the original execution if run directly
if __name__ == "__main__":
    # Use default folder if run directly
    default_folder = "models/cars"
    compare_models(default_folder)
