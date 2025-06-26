from ultralytics import YOLO
import cv2
import os

model = YOLO("best3.pt")

class_dict = {
    0: "Car",
    1: "Bike",
    2: "Van",
    3: "Bus",
    4: "Ambulance",
    5: "Airplane",
    6: "Truck"
}

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

def detect_objects(image_path: str):
    results = model(image_path)
    detections = []
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence > 0.3:
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
