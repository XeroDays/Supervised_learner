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

## Run Application

```bash
cd Supervised_learner
git pull
chmod +x start-linux.sh
./start-linux.sh
```

## Prepare Dataset

Put any `.zip` or `.7z` dataset archive into the `dataset/` folder. Do not extract it yourself.

Then run the script:

```bash
cd Supervised_learner
chmod +x start-linux.sh
./start-linux.sh
```

In the menu, select **1. Prepare Dataset (Extract Zip/7z)**. That extracts the archive, moves images and labels into `dataset/`, and deletes leftover empty folders. Zip/7z files are kept.

After that, `dataset/` should look like this:

```
dataset/
├── classes.txt
├── image1.jpg
├── image1.txt
├── image2.jpg
├── image2.txt
├── your-dataset.zip
└── ...
```

## GPU Troubleshooting (RTX 50-series / CUDA 12.8)

If training is slow or shows `pin_memory ... no accelerator is found`, PyTorch is running on CPU.

RTX 5060 Ti (Blackwell) requires PyTorch built with **CUDA 12.8**. Do **not** use `cu121` or `cu124`.

**Verify CUDA in the venv:**

```bash
source venv/bin/activate
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output:
- Version contains `+cu128` (e.g. `2.7.1+cu128`)
- `True`
- `NVIDIA GeForce RTX 5060 Ti`

**Manual fix:**

```bash
source venv/bin/activate
pip install --force-reinstall torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
```

**Full reset if still broken:**

```bash
rm -rf venv
./start-linux.sh
```

During training you should see a `GPU_mem` value (not `0G`) and no `pin_memory` warning.

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
