#!/bin/bash
# Supervisor Learner Linux Script
# Sets up venv, installs dependencies, runs main.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

fail() {
    echo -e "${RED}Error: $1${NC}"
    exit 1
}

verify_project_files() {
    local missing=()
    for file in Engine/__init__.py Engine/start.py Engine/compare.py Engine/train.py main.py; do
        if [ ! -f "$SCRIPT_DIR/$file" ]; then
            missing+=("$file")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        echo -e "${RED}Missing project files:${NC}"
        printf '  %s\n' "${missing[@]}"
        fail "Update the repo with: git pull"
    fi
}

install_pytorch_cuda() {
    if ! command -v nvidia-smi &> /dev/null; then
        return
    fi

    echo -e "${YELLOW}Installing PyTorch with CUDA 12.1 for GPU training...${NC}"
    "$VENV_DIR/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q \
        || echo -e "${YELLOW}Warning: CUDA PyTorch install failed. Training may run on CPU.${NC}"
}

check_gpu() {
    echo -e "\n${GREEN}=== GPU Check ===${NC}"

    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
    else
        echo -e "${YELLOW}Warning: nvidia-smi not found. Inference may run on CPU.${NC}"
    fi

    "$VENV_DIR/bin/python" - <<'PY'
import torch

if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available — will run on CPU")
PY
}

setup_venv() {
    echo -e "\n${GREEN}=== Environment Setup ===${NC}"

    PYTHON_CMD=""
    for candidate in python3.13 python3.12 python3.11 python3; do
        if command -v "$candidate" &> /dev/null; then
            PYTHON_CMD="$candidate"
            break
        fi
    done

    [ -n "$PYTHON_CMD" ] || fail "python3 is not installed. Run: apt install -y python3-venv python3-pip"

    PYTHON_VERSION=$("$PYTHON_CMD" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo -e "${YELLOW}Using $PYTHON_CMD (Python $PYTHON_VERSION)${NC}"

    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        "$PYTHON_CMD" -m venv "$VENV_DIR" || fail "Failed to create virtual environment"
    fi

    echo -e "${YELLOW}Installing dependencies...${NC}"
    "$VENV_DIR/bin/python" -m pip install --upgrade pip -q
    "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" -q || fail "Failed to install requirements"
    install_pytorch_cuda

    PYTHON_MINOR=$("$VENV_DIR/bin/python" -c 'import sys; print(sys.version_info.minor)')
    if [ "$PYTHON_MINOR" -ge 11 ] && [ -f "$SCRIPT_DIR/requirements-tflite.txt" ]; then
        echo -e "${YELLOW}Installing TFLite export dependencies...${NC}"
        "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements-tflite.txt" -q \
            || echo -e "${YELLOW}Warning: TFLite dependencies failed to install. Training will still work; export may be skipped.${NC}"
    else
        echo -e "${YELLOW}Python < 3.11 detected — skipping TFLite export dependencies.${NC}"
        echo -e "${YELLOW}GPU training works; export TFLite locally on Windows or use Python 3.11+.${NC}"
    fi

    echo -e "${GREEN}Environment ready.${NC}"
}

run_main() {
    cd "$SCRIPT_DIR" || fail "Could not enter project directory"
    echo -e "\n${GREEN}Starting main.py...${NC}\n"
    "$VENV_DIR/bin/python" main.py
}

clear
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}   Supervisor Learner${NC}"
echo -e "${CYAN}========================================${NC}"

command -v python3 &> /dev/null || fail "python3 is not installed. Run: apt install -y python3-venv python3-pip"

verify_project_files
setup_venv
check_gpu
run_main

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}main.py exited successfully.${NC}"
else
    echo -e "${YELLOW}main.py exited with code: $EXIT_CODE${NC}"
fi

echo ""
