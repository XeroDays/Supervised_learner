import os
import random
import shutil
import zipfile

import torch
import yaml
from ultralytics import YOLO

TRAIN_SPLIT = 0.90
DEFAULT_TRAIN_EPOCHS = 1500
IMG_SIZE = 640
BASE_MODEL = "yolov8n.yaml"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _project_root():
    return os.getcwd()


def _engine_dir():
    return os.path.dirname(os.path.abspath(__file__))


def prompt_epochs():
    while True:
        try:
            choice = input(
                f"\nEnter number of training epochs (default: {DEFAULT_TRAIN_EPOCHS}, press Enter to use default): "
            ).strip()
            if not choice:
                print(f"Using default: {DEFAULT_TRAIN_EPOCHS} epochs")
                return DEFAULT_TRAIN_EPOCHS
            epochs = int(choice)
            if epochs < 1:
                print("Please enter a positive number")
                continue
            return epochs
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            raise SystemExit(0)


def read_classes(dataset_path):
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

    return classes, classes_file


def _label_path_for_image(image_file):
    return image_file.rsplit(".", 1)[0] + ".txt"


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

    return paired_files


def prepare_comparer_dataset(dataset_path):
    paired_files = _pair_dataset_files(dataset_path)
    if not paired_files:
        raise ValueError("No paired image/label files found in dataset folder")

    comparer_path = os.path.join(_engine_dir(), "comparer")
    if os.path.exists(comparer_path):
        shutil.rmtree(comparer_path)
        print("Cleared existing comparer folder")

    train_images_dir = os.path.join(comparer_path, "images", "train")
    val_images_dir = os.path.join(comparer_path, "images", "val")
    train_labels_dir = os.path.join(comparer_path, "labels", "train")
    val_labels_dir = os.path.join(comparer_path, "labels", "val")

    for directory in (train_images_dir, val_images_dir, train_labels_dir, val_labels_dir):
        os.makedirs(directory, exist_ok=True)

    random.shuffle(paired_files)
    split_index = int(len(paired_files) * TRAIN_SPLIT)
    train_files = paired_files[:split_index]
    val_files = paired_files[split_index:]

    def copy_pairs(pairs, images_dir, labels_dir):
        for image_file, label_file in pairs:
            shutil.copy2(
                os.path.join(dataset_path, image_file),
                os.path.join(images_dir, image_file),
            )
            shutil.copy2(
                os.path.join(dataset_path, label_file),
                os.path.join(labels_dir, label_file),
            )

    copy_pairs(train_files, train_images_dir, train_labels_dir)
    copy_pairs(val_files, val_images_dir, val_labels_dir)

    print(f"Dataset prepared: {len(train_files)} train, {len(val_files)} val")
    return comparer_path


def write_data_yaml(classes):
    data_yaml_content = {
        "train": "comparer/images/train",
        "val": "comparer/images/val",
        "nc": len(classes),
        "names": classes,
    }

    data_yaml_path = os.path.join(_engine_dir(), "data.yaml")
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)

    print(f"Created data.yaml with {len(classes)} classes: {classes}")
    return data_yaml_path


def clear_output_folder():
    output_path = os.path.join(_project_root(), "output")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        print("Cleared output folder")
    os.makedirs(output_path, exist_ok=True)
    return output_path


def export_artifacts(best_weights_path, classes_file, output_dir):
    best_pt_dest = os.path.join(output_dir, "best.pt")
    shutil.copy2(best_weights_path, best_pt_dest)
    print(f"Exported: {best_pt_dest}")

    model = YOLO(best_weights_path)
    try:
        export_path = model.export(format="tflite", imgsz=IMG_SIZE)
        tflite_dest = os.path.join(output_dir, os.path.basename(export_path))
        if os.path.abspath(export_path) != os.path.abspath(tflite_dest):
            shutil.copy2(export_path, tflite_dest)
        print(f"Exported: {tflite_dest}")
    except Exception as e:
        print(f"Warning: TFLite export skipped ({e})")
        print("Install requirements-tflite.txt on Python 3.11+ to enable TFLite export.")

    classes_dest = os.path.join(output_dir, "classes.txt")
    shutil.copy2(classes_file, classes_dest)
    print(f"Exported: {classes_dest}")

    return best_pt_dest, tflite_dest, classes_dest


def zip_output_folder(output_dir, zip_name="output.zip"):
    zip_path = os.path.join(output_dir, zip_name)
    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for root, _, files in os.walk(output_dir):
            for file_name in files:
                if file_name == zip_name:
                    continue
                file_path = os.path.join(root, file_name)
                archive.write(file_path, os.path.relpath(file_path, output_dir))

    print(f"Created zip: {zip_path}")
    return zip_path


def train_model():
    dataset_path = os.path.join(_project_root(), "dataset")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    classes, classes_file = read_classes(dataset_path)
    epochs = prompt_epochs()
    prepare_comparer_dataset(dataset_path)
    data_yaml_path = write_data_yaml(classes)
    output_dir = clear_output_folder()

    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        device = 0
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("WARNING: Training on CPU — install PyTorch cu128 for RTX 5060 Ti")

    print(f"\nStarting training ({epochs} epochs, imgsz={IMG_SIZE})...")
    model = YOLO(BASE_MODEL)
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=IMG_SIZE,
        device=device,
        workers=8 if device != "cpu" else 0,
        project=output_dir,
        name="training",
        exist_ok=True,
    )

    best_weights_path = os.path.join(output_dir, "training", "weights", "best.pt")
    if not os.path.exists(best_weights_path):
        raise FileNotFoundError(f"Training completed but weights not found: {best_weights_path}")

    print("\nExporting model artifacts...")
    export_artifacts(best_weights_path, classes_file, output_dir)
    zip_output_folder(output_dir)
    print("\nTraining completed!")


if __name__ == "__main__":
    train_model()
