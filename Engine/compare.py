from ultralytics import YOLO
import pandas as pd
import os
import yaml

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
    return data_yaml_path

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
        import shutil
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
            import shutil
            shutil.copy2(src_path, dst_path)
            print(f"Copied image: {image_file}")
    
    # Copy text files to comparer/labels/val
    for text_file in text_files:
        src_path = os.path.join(dataset_path, text_file)
        dst_path = os.path.join(labels_val_path, text_file)
        if os.path.exists(src_path):
            import shutil
            shutil.copy2(src_path, dst_path)
            print(f"Copied label: {text_file}")
    
    print(f"Dataset setup complete. Files copied to: {comparer_path}")
    return comparer_path

def compare_models(folder_path):
    """Compare all models in the specified folder"""
    # Setup comparer dataset first
    print("Setting up comparer dataset...")
    comparer_path = setup_comparer_dataset()
    
    # Create data.yaml file
    data_yaml_path = create_data_yaml(folder_path)
    
    # Get all .pt files in the folder
    model_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
    if not model_files:
        print(f"No .pt model files found in {folder_path}")
        return
    
    # Create model paths dictionary
    model_paths = {}
    for model_file in sorted(model_files):
        model_name = f"Model {model_file.replace('.pt', '').replace('best', '')}"
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
    
    for model_name, model_path in model_paths.items():
        try:
            print(f"\nEvaluating {model_name}...")
            model = YOLO(model_path)
            results = model.val(data=data_yaml_path, split="val", verbose=False)
            metrics = results.results_dict
    
            # Extract only the metrics we care about, fallback to None if missing
            filtered_metrics = {
                metrics_to_extract[k]: metrics.get(k, None) for k in metrics_to_extract
            }
    
            results_summary[model_name] = filtered_metrics
    
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

# For backward compatibility, keep the original execution if run directly
if __name__ == "__main__":
    # Use default folder if run directly
    default_folder = "models/cars"
    compare_models(default_folder)
