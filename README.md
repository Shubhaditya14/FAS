# FAS - Face Anti-Spoofing System

A deep learning-based system for detecting face presentation attacks in biometric authentication systems.

## 🎯 Overview

This project implements a configurable Face Anti-Spoofing (FAS) system using PyTorch that can detect various spoofing attempts including:
- 📄 Printed photos
- 📱 Screen replays (phone/tablet displays)
- 🎭 Masks
- 📹 Video replays

## ✨ Features

- **Multiple Backbone Support**: EfficientNet, ResNet, MobileNet, and any timm model
- **Configuration-Driven**: All settings managed via YAML files
- **Comprehensive Metrics**: APCER, BPCER, ACER, EER, and standard ML metrics
- **Multi-Device Support**: MPS (Apple Silicon), CUDA, CPU
- **Data Augmentation**: Advanced augmentation pipeline using Albumentations
- **Web Interface**: Interactive Streamlit application
- **Real-time Inference**: Support for images, videos, and webcam
- **Experiment Tracking**: TensorBoard and WandB integration

## 📋 Requirements

- Python 3.11 or 3.12 (Python 3.13 has PyTorch compatibility issues)
- PyTorch 2.1+
- See `requirements.txt` for full dependencies

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd FAS

# Create virtual environment (use Python 3.11 or 3.12)
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Organize your dataset in the following structure:

```
data/splits/
├── train/
│   ├── real/
│   │   ├── img1.jpg
│   │   └── ...
│   └── spoof/
│       ├── img1.jpg
│       └── ...
├── val/
│   ├── real/
│   └── spoof/
└── test/
    ├── real/
    └── spoof/
```

### 3. Train Model

```bash
# Train with default configuration
python train.py

# Monitor training
tensorboard --logdir logs
```

### 4. Evaluate Model

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth \
                   --test-path data/splits/test \
                   --output-dir outputs/evaluation
```

### 5. Run Inference

```bash
# Single image
python inference.py --image path/to/image.jpg \
                    --checkpoint checkpoints/best_model.pth

# Video
python inference.py --video path/to/video.mp4 \
                    --checkpoint checkpoints/best_model.pth \
                    --display

# Webcam
python inference.py --camera 0 \
                    --checkpoint checkpoints/best_model.pth \
                    --display
```

### 6. Launch Web App

```bash
streamlit run app.py
```

## 📁 Project Structure

```
fas-system/
├── configs/              # YAML configuration files
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── inference_config.yaml
├── data/                 # Data directory
├── models/               # Model definitions
│   ├── backbones/        # Feature extractors
│   └── fusion/           # Fusion modules
├── utils/                # Utility functions
│   ├── data_loader.py
│   ├── augmentations.py
│   ├── metrics.py
│   └── visualization.py
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── inference.py          # Inference script
├── app.py                # Streamlit web app
└── requirements.txt      # Dependencies
```

## ⚙️ Configuration

Edit YAML files in `configs/` to customize:

- **Model**: Backbone architecture, fusion strategy, classifier settings
- **Training**: Learning rate, batch size, optimizer, scheduler, augmentations
- **Inference**: Preprocessing, thresholds, post-processing

## 📊 Metrics

The system calculates FAS-specific metrics:

- **APCER**: Attack Presentation Classification Error Rate
- **BPCER**: Bona Fide Presentation Classification Error Rate
- **ACER**: Average Classification Error Rate
- **EER**: Equal Error Rate

Plus standard metrics: Accuracy, Precision, Recall, F1, AUC-ROC

## 🎮 Usage Examples

### Custom Training Configuration

```bash
python train.py --model-config configs/custom_model.yaml \
                --train-config configs/custom_training.yaml
```

### Batch Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth \
                   --test-path data/splits/test \
                   --batch-size 64 \
                   --device mps
```

### Video Processing with Output

```bash
python inference.py --video input.mp4 \
                    --checkpoint checkpoints/best_model.pth \
                    --output results/output.mp4 \
                    --display
```

## 🔧 Development

See `CLAUDE.md` for detailed development guidelines and architecture documentation.

## 📈 Project Status

Current status tracked in `PROJECT_STATE.md`

**Completion: 35%**
- ✅ Environment setup
- ✅ Project structure
- ✅ Configuration system
- ✅ Core implementation
- ⏳ Dependency installation (PyTorch compatibility issue)
- ⏳ Data preparation
- ⏳ Model training
- ⏳ Evaluation
- ⏳ Deployment

## 🐛 Known Issues

1. **PyTorch Installation**: Python 3.13 compatibility issue - use Python 3.11 or 3.12
2. **MPS Mixed Precision**: Not yet supported by PyTorch for Apple Silicon

## 🤝 Contributing

1. Update appropriate configuration files
2. Follow existing code style and structure
3. Add comprehensive docstrings
4. Test thoroughly before committing
5. Update PROJECT_STATE.md with progress

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- timm library for model architectures
- Albumentations for data augmentation
- FAS research community for metrics and best practices
