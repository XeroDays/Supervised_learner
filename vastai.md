# Vast.ai Setup

## Generate SSH Key (Windows)

```powershell
ssh-keygen -t ed25519 -C "vast-ai"
type $env:USERPROFILE\.ssh\id_ed25519.pub
```

### Windows - Lenovo Laptop

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOV31m/yAx7W3D8N6RHt682E923cbwpj3Ktm3smbf6ee
```

### MacOs - Personal laptop

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIK8gCu9dkrbujaUn/17Z0A6tsIn+5I7CO4a1Re5k7r3b
```

---

## Clone Repository & Install Dependencies

```bash
git clone --depth 1 https://github.com/XeroDays/Supervised_learner.git
apt update
apt install -y python3-venv python3-pip
cd Supervised_learner
```

## Test GPU (CUDA dry run)

```bash
nvidia-smi
```

## Prepare Dataset

Place labeled images in `dataset/` (each image needs a matching `.txt` label file and a `classes.txt`):

```
dataset/
├── classes.txt
├── image1.jpg
├── image1.txt
├── image2.jpg
├── image2.txt
└── ...
```

Upload from local machine (run on your laptop):

```powershell
scp -P PORT -r dataset root@IP:/root/Supervised_learner/dataset
```

### Upload & Extract Zip Dataset

If your data is in a zip file named `Yolo_Vehicles_1280_clean.zip`, upload it into the `dataset/` folder and extract it on the server.

  
**2. Extract on the server:**

```bash
cd /root/Supervised_learner/dataset
apt install -y unzip
unzip Yolo_Vehicles_1280_clean.zip
```

**3. Move all images and labels into `dataset/` (if they are inside subfolders):**

```bash
cd /root/Supervised_learner/dataset
find . -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.txt" \) ! -name "classes.txt" -exec mv -n {} . \;
```

Your `dataset/` folder should look like this before training:

```
dataset/
├── classes.txt
├── image1.jpg
├── image1.txt
├── image2.jpg
├── image2.txt
└── ...
```

## Run Application

```bash
cd Supervised_learner
git pull
chmod +x start-linux.sh
./start-linux.sh
```

If you see `No module named 'Engine.compare'`, the server repo is out of date:

```bash
git pull origin main
ls Engine/compare.py
```

If the file is missing, push your latest code from your laptop first, then pull again on the server.

The script will:

- Create a virtual environment and install `requirements.txt`
- Check GPU availability
- Launch `main.py` (detection, model comparison, and training via the interactive menu)

## Download Trained Model

```powershell
scp -P PORT root@IP:/root/Supervised_learner/output/best.pt .
scp -P PORT root@IP:/root/Supervised_learner/output/best.tflite .
```

## Re-export TFLite on Linux (optional, requires Python 3.11+)

```bash
cd Supervised_learner
source venv/bin/activate
pip install -r requirements-tflite.txt
python -c "from ultralytics import YOLO; YOLO('output/best.pt').export(format='tflite', imgsz=640)"
```

---

## URL

https://vast.ai
