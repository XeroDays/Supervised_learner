 
## 🦾 YOLO Auto Pre-Labeling Tool

A powerful tool built on top of **Ultralytics YOLO** that automatically pre-labels images in your dataset to give you a head start on the annotation process. This software runs your dataset through a YOLO model, generates bounding boxes and YOLO-style annotation files (`.txt`), and prepares them for manual correction using tools like **LabelImg**.

---

### 🚀 Features

* 🔍 **Pre-label images using a YOLO model**
* 📂 **Read entire datasets before manual labeling**
* ✍️ **Generate YOLO `.txt` files for bounding boxes**
* 🎯 **Confidence thresholding** to control what gets labeled
* 📝 **Easily relabel manually for improved accuracy**
* 🧠 Works great as a **head start for training a custom model**

---
 

<img width="1461" height="633" alt="image" src="https://github.com/user-attachments/assets/ad586630-89d2-4dcd-8af4-dd1f405d0d48" />


![image](https://github.com/user-attachments/assets/d02cc218-b9b7-491f-a9df-1df0044e6e15)



---

## Precision & Accuracy

<img width="465" height="206" alt="image" src="https://github.com/user-attachments/assets/2a30fd5d-93cb-44ef-80b1-59213b49166d" />

<img width="534" height="319" alt="image" src="https://github.com/user-attachments/assets/65bab004-5293-4185-9768-c684885951e2" />


---

## 🔧 Requirements

* Python 3.13.2
* Ultralytics YOLO
* OpenCV

Install dependencies:

```bash
pip install ultralytics opencv-python
```

---

## 🚀 How to Use

1. Place your YOLO model file as `best.pt` in the root directory.
2. Add your test images to the `dataset/` folder.
3. Run the detection pipeline:

```bash
python start.py
```

This will:

* Detect objects in all images
* Draw and save bounding boxes to `saved/`
* Write YOLO `.txt` labels to `labels/`
* Create `classes.txt` in `labels/`

---

## 📁 Folder Structure

```
project/
├── dataset/            # Input folder containing images to detect
├── saved/              # Output folder for images with bounding boxes
├── labels/             # YOLO .txt files and classes.txt
├── best.pt             # Your trained YOLO model
├── start.py            # Main script to run everything
├── yolo_detection.py   # All detection logic
└── README.md           # This file
```

## ✨ Output Format

### Bounding Box Drawing

Draws bounding boxes on images with confidence scores, e.g.:

```
ID:0 (0.87)
```

### YOLO `.txt` Format

Each line:

```
<class_id> <x_center> <y_center> <width> <height>
```

Example:

```
0 0.433644 1.367841 0.131999 0.225675
```

### `classes.txt` Example

```
Car
Bike
Van
Bus
Ambulance
Airplane
Truck
```

---

## 🧠 Customization

* Update `save_classes_txt()` in `yolo_detection.py` to define your own class names.
* Change the confidence threshold inside `detect_objects()` if needed.
* Extend `start.py` to support video input, CSV logging, or batch evaluation.

---

## 🛡 License

MIT License

---

## 👨‍💻 Author

Built with ❤️ in Pakistan
