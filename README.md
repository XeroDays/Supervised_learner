# YOLO Object Detection Pipeline

This project is a lightweight object detection utility using the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework. It performs the following tasks:

* Detects objects in images using a YOLO `.pt` model
* Filters detections based on a confidence threshold
* Draws bounding boxes on the images
* Saves the output images in a `saved/` folder
* Generates YOLO-format `.txt` label files for training
* Outputs `classes.txt` containing the class names


## 📝 Description
This project helps you generate training data for retraining a YOLO model. It automatically creates YOLO-format .txt annotation files for each image based on initial detections. You can then review and correct these annotations using tools like LabelImg, and prepare an improved dataset to train the model again for better accuracy.
 

<img width="1461" height="633" alt="image" src="https://github.com/user-attachments/assets/ad586630-89d2-4dcd-8af4-dd1f405d0d48" />


![image](https://github.com/user-attachments/assets/d02cc218-b9b7-491f-a9df-1df0044e6e15)



---

## Precision & Accuracy

<img width="465" height="206" alt="image" src="https://github.com/user-attachments/assets/2a30fd5d-93cb-44ef-80b1-59213b49166d" />



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
