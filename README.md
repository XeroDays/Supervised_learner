# 🦾 YOLO Model Selection and Detection System

A comprehensive tool built on top of **Ultralytics YOLO** that provides both automatic image detection and model comparison capabilities. This software allows you to select from multiple trained models, process images with detection, and compare model performance across different datasets.

---

## 🚀 Features

* 🎯 **Dynamic Model Selection** - Choose from multiple model folders and files
* 🔍 **Automatic Image Detection** - Process images with selected YOLO models
* 📊 **Model Comparison** - Compare performance of multiple models
* 📂 **Organized Output** - All results saved in structured output folder
* 🎬 **Video Generation** - Create videos from detection results
* 🧠 **Dynamic Class Loading** - Automatically loads classes from model folders
* 📝 **YOLO Format Support** - Generates standard YOLO annotation files

---

## 📸 Screenshots

### Main Interface
<img width="1461" height="633" alt="Main Interface" src="https://github.com/user-attachments/assets/ad586630-89d2-4dcd-8af4-dd1f405d0d48" />

### Detection Results
![Detection Results](https://github.com/user-attachments/assets/d02cc218-b9b7-491f-a9df-1df0044e6e15)

---

## 🎯 Precision & Accuracy

### Detection Accuracy
<img width="465" height="206" alt="Detection Accuracy" src="https://github.com/user-attachments/assets/2a30fd5d-93cb-44ef-80b1-59213b49166d" />

### Model Performance
<img width="534" height="319" alt="Model Performance" src="https://github.com/user-attachments/assets/65bab004-5293-4185-9768-c684885951e2" />

---

## 🔧 Requirements

* Python 3.13.2
* Ultralytics YOLO
* OpenCV
* Pandas (for model comparison)

Install dependencies:

```bash
pip install ultralytics opencv-python pandas pyyaml
```

---

## 🚀 How to Use

### 0. Setup Virtual Environment

First, create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 1. Setup Your Models

Organize your trained models in the following structure:

```
models/
├── cars/
│   ├── best1.pt
│   ├── best2.pt
│   ├── best3.pt
│   └── classes.txt
└── receipts/
    ├── best1.pt
    └── classes.txt
```

### 2. Prepare Your Dataset

Place your images and labels in the `dataset/` folder:

```
dataset/
├── image1.jpg
├── image1.txt
├── image2.jpg
├── image2.txt
└── ...
```

### 3. Run the System

Make sure your virtual environment is activated, then run:

```bash
# Ensure virtual environment is activated (you should see (venv) in your terminal)
python main.py
```

---

## 📋 Workflow

### Step 1: Feature Selection
Choose between two main features:
- **Process Images (Detection)** - Run object detection on your dataset
- **Compare Models** - Compare performance of multiple models

### Step 2: Model Folder Selection
Select the model folder (e.g., "cars" or "receipts") containing your trained models.

### Step 3: Processing

#### For Detection:
1. Select specific model file (e.g., "best1.pt")
2. System loads model and classes automatically
3. Processes all images in dataset folder
4. Generates results in organized output structure

#### For Model Comparison:
1. System automatically finds all models in selected folder
2. Creates comparer dataset structure
3. Evaluates each model on validation data
4. Generates comparison report

---

## 📁 Project Structure

```
project/
├── main.py                 # Main entry point
├── Engine/                 # Core engine files
│   ├── start.py           # Detection processing
│   ├── yolo_detection.py  # YOLO detection logic
│   ├── create_video.py    # Video generation
│   ├── compare.py         # Model comparison
│   ├── data.yaml          # Dynamic data configuration
│   └── comparer/          # Comparison dataset (auto-created)
│       ├── images/val/    # Validation images
│       └── labels/val/    # Validation labels
├── models/                # Your trained models
│   ├── cars/
│   │   ├── best1.pt
│   │   ├── best2.pt
│   │   └── classes.txt
│   └── receipts/
│       ├── best1.pt
│       └── classes.txt
├── dataset/               # Input images and labels
├── output/                # All generated results
│   ├── saved/            # Images with detections
│   ├── labels/           # YOLO annotation files
│   ├── output_video.mp4  # Generated video
│   └── model_comparison.csv # Comparison results
└── README.md
```

---

## ✨ Output Format

### Detection Results

#### Images with Bounding Boxes
- Saved in `output/saved/`
- Shows detected objects with confidence scores
- Color-coded by class

#### YOLO Annotation Files
- Saved in `output/labels/`
- Standard YOLO format: `<class_id> <x_center> <y_center> <width> <height>`
- Normalized coordinates (0-1)

#### Classes File
- `output/labels/classes.txt`
- Lists all class names used by the model

### Model Comparison Results

#### CSV Report
- Saved as `output/model_comparison.csv`
- Contains metrics: Precision, Recall, mAP50, mAP50-95
- Easy to analyze in Excel or other tools

---

## 🎯 Example Workflow

### Detection Example:
```
1. Run: python main.py
2. Select: "1. Process Images (Detection)"
3. Select: "1. cars" (model folder)
4. Select: "1. best1.pt" (specific model)
5. Results: Images with detections in output/saved/
```

### Comparison Example:
```
1. Run: python main.py
2. Select: "2. Compare Models"
3. Select: "1. cars" (model folder)
4. Results: Comparison report in output/model_comparison.csv
```

---

## 🧠 Dynamic Features

### Automatic Class Loading
- Reads `classes.txt` from model folder
- Automatically configures detection classes
- No manual class configuration needed

### Dynamic Data Configuration
- Creates `data.yaml` automatically for model comparison
- Uses classes from selected model folder
- Configures validation paths correctly

### Organized Output
- All results saved in `output/` folder
- Automatic folder creation and cleanup
- Consistent file organization

---

## 🛡 Error Handling

The system includes comprehensive error handling:
- Model validation before processing
- Class file verification
- Dataset folder existence checks
- Clear error messages for troubleshooting

---

## 🛡 License

MIT License

---

## 👨‍💻 Author

Built with ❤️ in Pakistan