#!/bin/bash

# FAS Portable Packager
# Creates a portable distribution with only necessary files

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║     FAS Portable Packager                              ║"
echo "║     Create portable distribution package               ║"
echo "║                                                           ║"
echo "╚═════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if we're in the right directory
if [ ! -f "app_webcam.py" ] || [ ! -f "inference.py" ]; then
    echo -e "${RED}Error: Run this script from the FAS project root${NC}"
    exit 1
fi

# Create temporary directory for packaging
TEMP_DIR="fas_portable"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

echo -e "${CYAN}📦 Creating portable package...${NC}"

# Create directory structure in temp
mkdir -p "$TEMP_DIR/models"
mkdir -p "$TEMP_DIR/utils"
mkdir -p "$TEMP_DIR/configs"

# Copy essential files
echo -e "${GREEN}  Copying Python files...${NC}"
cp app_webcam.py "$TEMP_DIR/"
cp inference.py "$TEMP_DIR/"
cp multi_model_predictor.py "$TEMP_DIR/"
cp utils/*.py "$TEMP_DIR/utils/" 2>/dev/null || true
cp models/*.py "$TEMP_DIR/models/" 2>/dev/null || true

# Copy models
echo -e "${GREEN}  Copying model weights...${NC}"
if [ -d "pth" ]; then
    cp pth/*.pth "$TEMP_DIR/models/" 2>/dev/null || true
fi

# Copy configs
echo -e "${GREEN}  Copying configuration files...${NC}"
if [ -d "configs" ]; then
    cp configs/*.yaml "$TEMP_DIR/configs/" 2>/dev/null || true
fi

# Copy requirements
echo -e "${GREEN}  Copying requirements...${NC}"
if [ -f "requirements.txt" ]; then
    cp requirements.txt "$TEMP_DIR/"
fi

# Create simple README
echo -e "${GREEN}  Creating README...${NC}"
cat > "$TEMP_DIR/README.md" << 'EOF'
# Face Anti-Spoofing (FAS) - Portable Version

Real-time face liveness detection using deep learning.

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or 3.12
- Virtual environment recommended
- Webcam/camera
- 4GB RAM minimum

### Installation

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎥 Running the Application

### Option 1: Live Webcam Detection (Recommended)

```bash
streamlit run app_webcam.py
```

This will:
- Open your browser
- Request camera permission
- Show continuous live video
- Classify faces in real-time (25-30 FPS)
- Display ✅ Real or ⚠️ Spoof with confidence

### Option 2: Process Images

```bash
python inference.py --image path/to/face.jpg --device mps
```

### Option 3: Process Videos

```bash
python inference.py --video path/to/video.mp4 --device mps --temporal-smoothing
```

## 📦 What's Included

### Essential Files
- `app_webcam.py` - Main webcam application
- `inference.py` - Inference engine
- `multi_model_predictor.py` - Ensemble support
- `models/` - FeatherNet architecture
- `utils/` - Helper functions
- `configs/` - Configuration files

### Pretrained Models (in models/)
- `AntiSpoofing_bin_128.pth` - Binary classifier (2.8 MB)
- `AntiSpoofing_print-replay_128.pth` - Print/replay model (2.8 MB)
- `AntiSpoofing_print-replay_1.5_128.pth` - Enhanced model (2.8 MB)

All models: 695K parameters, 128x128 input

## ⚙️ Configuration

Edit files in `configs/` to customize:
- Model architecture
- Inference settings
- Training parameters (if you add training)

## 📊 Features

### Detection
- Real-time face liveness detection
- Multiple model support (ensemble)
- Temporal smoothing for video
- Configurable detection threshold
- FPS tracking

### Supported Attacks
- 📄 Printed photos
- 📱 Screen replays (phone/tablet)
- 🎭 Masks
- 📹 Video replays

## 🎯 Usage

### Using Webcam
1. Run `streamlit run app_webcam.py`
2. Configure settings in sidebar (device, model, threshold)
3. Click "▶️ Start Detection"
4. Allow camera permission
5. See real-time classification results

### Understanding Results

- **✅ REAL (Green)** = Live person face detected
- **⚠️ SPOOF (Red)** = Spoof attack detected
- **Confidence** = How confident the model is (0-100%)
- **FPS** = Frames per second (performance metric)

### Troubleshooting

**Camera not working?**
- Check browser permissions
- Try Chrome instead of Safari
- Check if another app is using the camera

**Everything shows as Spoof?**
- Lower the detection threshold (in sidebar)
- Adjust lighting
- Try different model (if available)

**Slow performance?**
- Use MPS (Mac) or CUDA (GPU) instead of CPU
- Close other applications
- Check RAM usage

## 📝 Known Limitations

### Domain Shift
The pretrained models may have reduced accuracy on your specific camera/environment:
- Models were trained on CelebA-Spoof dataset
- Different lighting/conditions may affect performance
- This is a known data science challenge

**Solution:** For production use, fine-tune models on your specific data.

### Performance
- CPU: 10-15 FPS (slow but works everywhere)
- MPS (Mac): 25-30 FPS (recommended)
- CUDA (GPU): 30+ FPS (fastest)

## 📚 Documentation

For detailed technical information, refer to the full project documentation:
- Model architecture: See source code comments
- Metrics: See `utils/metrics.py`
- Configuration: See files in `configs/`

## 🤝 Development

This is a portable distribution for easy deployment and testing.
For development or modifications, use the full project repository.

---

## System Requirements

- **OS:** macOS 10.15+, Windows 10+, or Linux
- **Python:** 3.11 or 3.12 (not 3.13)
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 200MB for portable package
- **Camera:** Built-in or USB webcam

## License

This project is for educational and research purposes.
EOF

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    echo -e "${YELLOW}  Creating requirements.txt...${NC}"
    cat > "$TEMP_DIR/requirements.txt" << 'REQUIREMENTS'
# Core dependencies
torch>=2.1.0
torchvision>=0.16.0
Pillow>=10.0.0
numpy>=1.24.0

# UI
streamlit>=1.28.0

# Computer vision
opencv-python>=4.8.0
albumentations>=1.3.0

# Utilities
PyYAML>=6.0
REQUIREMENTS
fi

# Create minimal config files if needed
if [ ! -d "$TEMP_DIR/configs" ]; then
    mkdir -p "$TEMP_DIR/configs"
fi

# Create a simple model config if it doesn't exist
cat > "$TEMP_DIR/configs/inference_config.yaml" << 'YAML'
# Inference configuration for FAS system
model:
  name: FeatherNet
  num_classes: 2

input:
  image_size: [128, 128]
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]

device:
  name: mps  # Options: mps, cuda, cpu
  fallback: cpu

inference:
  confidence_threshold: 0.5
  batch_size: 1
  temporal_smoothing: true
  smoothing_window: 5
YAML
fi

# Check what we've created
echo ""
echo -e "${CYAN}📦 Files included:${NC}"
echo ""

# Python files
echo -e "${GREEN}Python Files:${NC}"
ls -lh "$TEMP_DIR"/*.py 2>/dev/null | grep -v "^total"
echo ""

# Models
echo -e "${GREEN}Model Weights:${NC}"
ls -lh "$TEMP_DIR/models/"/*.pth 2>/dev/null | grep -v "^total"
echo ""

# Configs
echo -e "${GREEN}Configuration:${NC}"
ls -lh "$TEMP_DIR/configs/" 2>/dev/null | grep -v "^total"
echo ""

# Utils
echo -e "${GREEN}Utilities:${NC}"
ls -lh "$TEMP_DIR/utils/"/*.py 2>/dev/null | grep -v "^total"
echo ""

# README
echo -e "${GREEN}Documentation:${NC}"
ls -lh "$TEMP_DIR/README.md" 2>/dev/null | grep -v "^total"
echo ""

# Total size
TOTAL_SIZE=$(du -sh "$TEMP_DIR" | cut -f1)
echo -e "${YELLOW}Total package size: $TOTAL_SIZE${NC}"
echo ""

# Create zip file
ZIP_NAME="fas-portable.zip"
echo -e "${CYAN}📦 Creating portable package...${NC}"
echo -e "  Package: ${ZIP_NAME}"

cd "$TEMP_DIR"
zip -r "../$ZIP_NAME" * 2>/dev/null
cd ..

# Cleanup temp directory
rm -rf "$TEMP_DIR"

# Verify zip was created
if [ -f "$ZIP_NAME" ]; then
    ZIP_SIZE=$(ls -lh "$ZIP_NAME" | awk '{print $5}')
    echo -e "${GREEN}✅ Package created: $ZIP_NAME ($ZIP_SIZE)${NC}"
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════${NC}"
    echo -e "${CYAN}📋 What's in the package:${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════${NC}"
    echo ""
    echo -e "${GREEN}✅ Ready to share!${NC}"
    echo ""
    echo -e "${CYAN}Quick instructions for recipient:${NC}"
    echo ""
    echo -e "${YELLOW}1. Extract the ZIP file${NC}"
    echo -e "   unzip $ZIP_NAME"
    echo -e "   cd fas-portable"
    echo ""
    echo -e "${YELLOW}2. Create virtual environment${NC}"
    echo -e "   python3 -m venv venv"
    echo -e "   source venv/bin/activate  # Mac/Linux"
    echo -e "   venv\Scripts\activate  # Windows"
    echo ""
    echo -e "${YELLOW}3. Install dependencies${NC}"
    echo -e "   pip install -r requirements.txt"
    echo ""
    echo -e "${YELLOW}4. Run the app${NC}"
    echo -e "   streamlit run app_webcam.py"
    echo ""
    echo -e "${YELLOW}5. Configure in browser${NC}"
    echo -e "   - Select device (MPS recommended for Mac)"
    echo -e "   - Choose model (AntiSpoofing_bin_128.pth recommended)"
    echo -e "   - Adjust threshold if needed (0.5 = balanced)"
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${GREEN}📦 Package contents:${NC}"
    echo ""
    unzip -l "$ZIP_NAME"
    echo ""
    echo -e "${CYAN}For more details, see README.md in the extracted folder${NC}"
    echo ""
else
    echo -e "${RED}❌ Error: Failed to create package${NC}"
    exit 1
fi
