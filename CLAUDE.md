# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FAS (Face Anti-Spoofing System)** is a deep learning-based system for detecting face presentation attacks (spoofing attempts) in biometric authentication systems. The project uses PyTorch and supports multiple backbone architectures (EfficientNet, ResNet, MobileNet) for binary classification of real vs. spoof faces.

## Technology Stack

- **Language:** Python 3.11+ (3.13 has PyTorch compatibility issues)
- **Deep Learning:** PyTorch, torchvision
- **Computer Vision:** OpenCV, Albumentations
- **Models:** timm, efficientnet-pytorch
- **Visualization:** matplotlib, tensorboard, wandb
- **Web Interface:** Streamlit
- **Configuration:** YAML (PyYAML)
- **Device Support:** MPS (Apple Silicon), CUDA, CPU

## Project Structure

```
fas-system/
├── configs/              # YAML configuration files
│   ├── model_config.yaml       # Model architecture settings
│   ├── training_config.yaml    # Training hyperparameters
│   └── inference_config.yaml   # Inference settings
├── data/                 # Data storage
│   ├── raw/                    # Original datasets
│   ├── processed/              # Preprocessed data
│   └── splits/                 # Train/val/test splits
├── models/               # Model definitions
│   ├── backbones/              # Feature extractors (EfficientNet, etc.)
│   ├── fusion/                 # Feature fusion modules
│   └── pretrained/             # Pretrained weights
├── utils/                # Utility modules
│   ├── data_loader.py          # Dataset and DataLoader
│   ├── augmentations.py        # Data augmentation pipelines
│   ├── metrics.py              # Metrics (APCER, BPCER, ACER, EER)
│   └── visualization.py        # Plotting and visualization
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── inference.py          # Inference script
├── app.py                # Streamlit web application
└── requirements.txt      # Python dependencies
```

## Common Commands

### Environment Setup
```bash
# Create virtual environment (use Python 3.11 or 3.12 for best compatibility)
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Train with default configs
python train.py

# Train with custom configs
python train.py --model-config configs/model_config.yaml \
                --train-config configs/training_config.yaml

# Monitor with TensorBoard
tensorboard --logdir logs
```

### Evaluation
```bash
# Evaluate on test set
python evaluate.py --checkpoint checkpoints/best_model.pth \
                   --test-path data/splits/test \
                   --output-dir outputs/evaluation

# Specify device
python evaluate.py --checkpoint checkpoints/best_model.pth \
                   --device mps  # or cuda, cpu
```

### Inference
```bash
# Single image inference
python inference.py --image path/to/image.jpg \
                    --checkpoint checkpoints/best_model.pth

# Video inference
python inference.py --video path/to/video.mp4 \
                    --checkpoint checkpoints/best_model.pth \
                    --output outputs/result.mp4 \
                    --display

# Webcam inference
python inference.py --camera 0 \
                    --checkpoint checkpoints/best_model.pth \
                    --display
```

### Web Application
```bash
# Launch Streamlit app
streamlit run app.py

# Specify port
streamlit run app.py --server.port 8501
```

## Architecture

### Data Pipeline
1. **FASDataset** (`utils/data_loader.py`): Custom PyTorch Dataset expecting `real/` and `spoof/` subdirectories
2. **Augmentations** (`utils/augmentations.py`): Albumentations-based transforms with:
   - Training: resize, flip, rotation, color jitter, noise, distortions
   - Validation/Test: resize + normalize only
3. **DataLoader**: Supports multi-worker loading, pinned memory for GPU transfer

### Model Architecture
- **Backbone**: Configurable feature extractor (default: EfficientNet-B0)
  - Supports: EfficientNet (B0-B7), ResNet, MobileNetV3, any timm model
  - Optional layer freezing for transfer learning
- **Fusion Module**: Combines features for classification
  - Types: concat, attention-based, adaptive
  - Configurable hidden layers and dropout
- **Classifier**: Binary classification head (Real vs Spoof)

### Training Pipeline
1. Load configs from YAML files
2. Create data loaders with augmentations
3. Initialize model, optimizer, scheduler
4. Training loop with:
   - Progress tracking (tqdm)
   - Metric calculation per epoch
   - TensorBoard/WandB logging
   - Checkpointing (best + periodic)
   - Early stopping
5. Generate training curves and save best model

### Metrics
FAS-specific metrics implemented in `utils/metrics.py`:
- **APCER**: Attack Presentation Classification Error Rate (false acceptance of spoofs)
- **BPCER**: Bona Fide Presentation Classification Error Rate (false rejection of real)
- **ACER**: Average Classification Error Rate (average of APCER and BPCER)
- **EER**: Equal Error Rate (where false acceptance = false rejection)
- Standard: Accuracy, Precision, Recall, F1, AUC-ROC

## Configuration System

All major settings are controlled via YAML configs in `configs/`:

### model_config.yaml
- Backbone architecture selection
- Feature fusion strategy
- Layer freezing for transfer learning
- Multi-modal settings (RGB, depth, IR)

### training_config.yaml
- Data paths and preprocessing
- Training hyperparameters (lr, batch size, epochs)
- Optimizer and scheduler settings
- Loss function configuration
- Regularization (dropout, early stopping)
- Device configuration (MPS/CUDA/CPU)
- Logging and checkpointing

### inference_config.yaml
- Model checkpoint path
- Input preprocessing
- Confidence thresholds
- Real-time inference settings

## Development Workflow

### Adding New Features
1. Update appropriate config YAML if needed
2. Implement feature in relevant module
3. Update this CLAUDE.md if it changes workflow
4. Test thoroughly before committing
5. Update PROJECT_STATE.md with progress

### Data Preparation
Expected directory structure for datasets:
```
data/splits/train/
├── real/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── spoof/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

Same structure for `val/` and `test/` directories.

### Model Development
- Backbones: Add to `models/backbones/__init__.py`
- Fusion modules: Add to `models/fusion/__init__.py`
- Always maintain config-driven approach
- Support both pretrained and from-scratch training

### Checkpointing
Checkpoints saved in `checkpoints/` contain:
- `epoch`: Training epoch number
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `metrics`: Performance metrics

Best model saved as `checkpoints/best_model.pth`

## Important Notes

### Device Compatibility
- **MPS (Apple Silicon)**: Supported but mixed precision not available yet
- **CUDA**: Full support including mixed precision training
- **CPU**: Fallback option, slower for training

### Python Version
- **Recommended**: Python 3.11 or 3.12
- **Not Recommended**: Python 3.13 (PyTorch compatibility issues as of Jan 2024)
- If using Python 3.13, recreate venv with 3.11/3.12

### Data Augmentation
- Training uses aggressive augmentation to improve generalization
- Validation/test use minimal preprocessing (resize + normalize)
- Test-time augmentation available in `utils/augmentations.py`

### Logging
- TensorBoard: Logs saved to `logs/` directory
- WandB: Optional, configure in `training_config.yaml`
- Training curves automatically saved after training

## Common Issues

1. **PyTorch installation fails**: Check Python version (use 3.11 or 3.12)
2. **MPS device errors**: Ensure macOS is up to date, fallback to CPU if issues persist
3. **Out of memory**: Reduce batch size in `training_config.yaml`
4. **Data not found**: Verify directory structure matches expected format
5. **Model convergence issues**: Check learning rate, try different optimizers

## Code Style

- Follow PEP 8 conventions
- Use type hints where appropriate
- Comprehensive docstrings for all functions/classes
- Configuration over hardcoding
- Modular, reusable components

## Git Workflow

- Commit frequently with descriptive messages
- Update PROJECT_STATE.md with each major milestone
- Use .gitignore to exclude checkpoints, data, logs
- Keep configs in version control
