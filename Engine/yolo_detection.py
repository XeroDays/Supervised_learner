from ultralytics import YOLO
import cv2
import os

# Global model variable
model = None
class_dict = {0: "Transaction1"}  # Default fallback

def set_model_path(model_path):
    """Set the model path and load the model"""
    global model, class_dict
    model = YOLO(model_path)
    print(f"Model loaded from: {model_path}")
    
    # Load classes from classes.txt in the model's folder
    model_dir = os.path.dirname(model_path)
    classes_file = os.path.join(model_dir, "classes.txt")
    
    if os.path.exists(classes_file):
        class_dict = {}
        with open(classes_file, 'r') as f:
            for idx, line in enumerate(f):
                class_name = line.strip()
                if class_name:  # Skip empty lines
                    class_dict[idx] = class_name
        print(f"Loaded {len(class_dict)} classes from {classes_file}")
        print(f"Classes: {list(class_dict.values())}")
    else:
        print(f"Warning: classes.txt not found in {model_dir}, using default class")
        class_dict = {0: "Transaction1"}

def check_model_loaded():
    """Check if model is loaded, raise error if not"""
    if model is None:
        raise RuntimeError("Error: No model selected. Please select a model first.")


# Optional: define distinct BGR colors for classes
class_colors = {
    0: (255, 0, 0),      # Blue
    1: (0, 255, 0),      # Green
    2: (0, 0, 255),      # Red
    3: (255, 255, 0),    # Cyan
    4: (255, 0, 255),    # Magenta
    5: (0, 255, 255),    # Yellow
    6: (128, 0, 128)     # Purple
}



def delete_txt_files(file_list):
    for file in file_list:
        if file.endswith(".txt"):
            os.remove(file)




def detect_objects(image_path: str):
    check_model_loaded()  # Check if model is loaded before proceeding
    results = model(image_path)
    detections = []
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence > 0.29:
                detections.append({
                    "class_id": int(box.cls[0]),
                    "confidence": confidence,
                    "bounding_box": box.xyxy[0].tolist()
                })
    return detections

def draw_and_save_detections(image_path: str, detections: list):
    img = cv2.imread(image_path)
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bounding_box'])
        class_id = det['class_id']
        class_name = class_dict.get(class_id, f"Class_{class_id}")
        label = f"{class_name} ({det['confidence']:.2f})"
        color = class_colors.get(class_id, (255, 255, 255))  # Default white if not found
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_dir = os.path.join(os.getcwd(), "saved")
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, file_name)
    cv2.imwrite(output_path, img)

def save_yolo_txt(image_path: str, detections: list):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    label_dir = os.path.join(os.getcwd(), "labels")
    os.makedirs(label_dir, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    output_path = os.path.join(label_dir, file_name)
 
    with open(output_path, "w") as f:
        for det in detections:
            x1, y1, x2, y2 = det["bounding_box"]
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h 
            f.write(f"{det['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

def save_classes_txt():
    label_dir = os.path.join(os.getcwd(), "labels")
    os.makedirs(label_dir, exist_ok=True)
    classes_path = os.path.join(label_dir, "classes.txt")
    with open(classes_path, "w") as f:
        for class_id in sorted(class_dict.keys()):
            f.write(f"{class_dict[class_id]}\n")
