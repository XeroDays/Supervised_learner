from ultralytics import YOLO
import pandas as pd

# Define paths to models and dataset
model_paths = {  
    "Model 4": "best4.pt",
    "Model 8": "best8.pt",
    "Model 9": "best9.pt",
    "Model 10": "best10.pt",
}
data_yaml = "data.yaml"

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
        results = model.val(data=data_yaml, split="val", verbose=False)
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
