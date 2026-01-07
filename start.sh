#!/bin/bash

# FAS Project Launcher Script
# Provides easy access to all project components

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Banner
echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║         Face Anti-Spoofing System (FAS)                  ║"
echo "║         Real-time Liveness Detection                     ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check virtual environment
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source venv/bin/activate

# Check dependencies
if ! python -c "import streamlit" 2>/dev/null; then
    echo -e "${YELLOW}📦 Installing dependencies...${NC}"
    pip install -q -r requirements.txt
    echo -e "${GREEN}✅ Dependencies installed${NC}"
fi

# Check for models
if [ ! -d "pth" ] || [ -z "$(ls -A pth/*.pth 2>/dev/null)" ]; then
    echo -e "${RED}⚠️  Warning: No models found in pth/ directory${NC}"
    echo -e "${YELLOW}Please download pretrained models and place them in pth/${NC}"
fi

# Menu
echo ""
echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}Select an option:${NC}"
echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}📹 LIVE DETECTION${NC}"
echo -e "  ${YELLOW}1)${NC} Continuous Live Detection (Real-time Video) ⭐"
echo -e "  ${YELLOW}2)${NC} Image Upload Interface"
echo -e "  ${YELLOW}3)${NC} Ensemble Demo"
echo ""
echo -e "${GREEN}🎬 VIDEO/IMAGE PROCESSING${NC}"
echo -e "  ${YELLOW}4)${NC} Process Single Image"
echo -e "  ${YELLOW}5)${NC} Process Video File"
echo -e "  ${YELLOW}6)${NC} Real-time Webcam (Terminal)"
echo ""
echo -e "${GREEN}🔬 EVALUATION & TESTING${NC}"
echo -e "  ${YELLOW}7)${NC} Evaluate Ensemble on Dataset"
echo -e "  ${YELLOW}8)${NC} Run Test Suite"
echo ""
echo -e "${GREEN}⚙️  UTILITIES${NC}"
echo -e "  ${YELLOW}9)${NC} List Available Models"
echo -e "  ${YELLOW}10)${NC} Check System Info"
echo -e "  ${YELLOW}11)${NC} Open Python Shell"
echo ""
echo -e "  ${YELLOW}0)${NC} Exit"
echo ""
echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"
echo -n -e "${CYAN}Enter your choice: ${NC}"
read choice

case $choice in
    1)
        echo -e "${GREEN}🎥 Launching Webcam Live Detection...${NC}"
        echo -e "${BLUE}Opening browser at http://localhost:8501${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        streamlit run app_webcam.py
        ;;
    
    2)
        echo -e "${GREEN}📸 Launching Image Upload Interface...${NC}"
        echo -e "${BLUE}Opening browser at http://localhost:8501${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        streamlit run app.py
        ;;
    
    3)
        echo -e "${GREEN}🔀 Launching Ensemble Demo...${NC}"
        echo -e "${BLUE}Opening browser at http://localhost:8501${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        streamlit run app_ensemble.py
        ;;
    
    4)
        echo -e "${CYAN}Enter image path: ${NC}"
        read image_path
        if [ ! -f "$image_path" ]; then
            echo -e "${RED}❌ Image not found: $image_path${NC}"
            exit 1
        fi
        echo -e "${GREEN}🖼️  Processing image...${NC}"
        python inference.py --image "$image_path" --device mps --display
        ;;
    
    5)
        echo -e "${CYAN}Enter video path: ${NC}"
        read video_path
        if [ ! -f "$video_path" ]; then
            echo -e "${RED}❌ Video not found: $video_path${NC}"
            exit 1
        fi
        echo -e "${CYAN}Enter output path (optional, press Enter to skip): ${NC}"
        read output_path
        
        if [ -z "$output_path" ]; then
            echo -e "${GREEN}🎬 Processing video...${NC}"
            python inference.py --video "$video_path" --device mps --display
        else
            echo -e "${GREEN}🎬 Processing video with output...${NC}"
            python inference.py --video "$video_path" --output "$output_path" --device mps --display
        fi
        ;;
    
    6)
        echo -e "${GREEN}📹 Starting real-time webcam detection...${NC}"
        echo -e "${YELLOW}Press 'q' to quit, 'space' to pause${NC}"
        python inference.py --camera 0 --device mps --temporal-smoothing --display
        ;;
    
    7)
        echo -e "${CYAN}Enter real images directory: ${NC}"
        read real_dir
        echo -e "${CYAN}Enter spoof images directory: ${NC}"
        read spoof_dir
        
        if [ ! -d "$real_dir" ] || [ ! -d "$spoof_dir" ]; then
            echo -e "${RED}❌ Directory not found${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}📊 Evaluating ensemble...${NC}"
        python eval_ensemble.py \
            --real-dir "$real_dir" \
            --spoof-dir "$spoof_dir" \
            --fusion-type all \
            --device mps \
            --output-dir eval_results
        ;;
    
    8)
        echo -e "${GREEN}🧪 Running test suite...${NC}"
        if [ -f "run_all_tests.sh" ]; then
            bash run_all_tests.sh
        else
            echo -e "${YELLOW}Running available tests...${NC}"
            [ -f "test_architecture.py" ] && python test_architecture.py
            [ -f "test_datasets.py" ] && python test_datasets.py
            [ -f "test_preprocessing.py" ] && python test_preprocessing.py
            [ -f "test_inference.py" ] && python test_inference.py
            echo -e "${GREEN}✅ Tests completed${NC}"
        fi
        ;;
    
    9)
        echo -e "${GREEN}📦 Available Models:${NC}"
        echo ""
        if [ -d "pth" ]; then
            for model in pth/*.pth pth/*.pth.tar; do
                if [ -f "$model" ] 2>/dev/null; then
                    size=$(du -h "$model" | cut -f1)
                    echo -e "  ${CYAN}•${NC} $(basename "$model") ${YELLOW}($size)${NC}"
                fi
            done
        else
            echo -e "${RED}No models found in pth/ directory${NC}"
        fi
        echo ""
        ;;
    
    10)
        echo -e "${GREEN}💻 System Information:${NC}"
        echo ""
        echo -e "${CYAN}Python Version:${NC}"
        python --version
        echo ""
        echo -e "${CYAN}PyTorch Version:${NC}"
        python -c "import torch; print(torch.__version__)"
        echo ""
        echo -e "${CYAN}Device Support:${NC}"
        python -c "import torch; print('MPS Available:', torch.backends.mps.is_available()); print('CUDA Available:', torch.cuda.is_available())"
        echo ""
        echo -e "${CYAN}Installed Packages:${NC}"
        pip list | grep -E "torch|streamlit|opencv|PIL|numpy"
        echo ""
        ;;
    
    11)
        echo -e "${GREEN}🐍 Opening Python shell...${NC}"
        echo -e "${YELLOW}Importing common modules...${NC}"
        python -c "
import sys
sys.path.insert(0, '.')
from models.feathernet import create_feathernet
from inference import FASInference
from multi_model_predictor import MultiModelPredictor
import torch
print('Loaded: create_feathernet, FASInference, MultiModelPredictor, torch')
print('Available device:', 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
" && python
        ;;
    
    0)
        echo -e "${GREEN}👋 Goodbye!${NC}"
        exit 0
        ;;
    
    *)
        echo -e "${RED}❌ Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✅ Done!${NC}"
