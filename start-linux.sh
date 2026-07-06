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

    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv "$VENV_DIR" || fail "Failed to create virtual environment"
    fi

    echo -e "${YELLOW}Installing dependencies...${NC}"
    "$VENV_DIR/bin/python" -m pip install --upgrade pip -q
    "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" -q || fail "Failed to install requirements"

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
