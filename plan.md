# FAS Project - Implementation Plan

**Created:** 2026-01-05  
**Goal:** Build ensemble model with cross-dataset generalization + Streamlit UI with webcam detection

---

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| FeatherNet Architecture | Done | 695K params, fully tested |
| Pretrained Weights | Done | 4 models available |
| Datasets | Done | OULU (1,700) + SIW (7,586 images) |
| Preprocessing | Done | Augmentations, TTA ready |
| Inference (FASPredictor) | Done | Single/batch/video support |
| Metrics | Done | APCER, BPCER, ACER, EER |
| Training Pipeline | Partial | Skeleton exists |
| Fusion Modules | Partial | Code exists, not integrated |
| Streamlit App | Partial | Basic skeleton only |

**Key Problem:** Cross-dataset performance is poor (OULU: 65%, SIW: 20%). Domain shift needs fixing.

---

## Plan Overview

| Step | Description | Status |
|------|-------------|--------|
| Step 1 | Ensemble Architecture | Pending |
| Step 2 | Domain Adaptation | Pending |
| Step 3 | Training Pipeline Integration | Pending |
| Step 4 | Evaluation & Testing | Pending |
| Step 5 | Inference API Refinement | Pending |
| Step 6 | Model Export (Optional) | Pending |
| Step 7 | Streamlit Frontend UI | Pending |

---

## Step 1: Ensemble Architecture

**Goal:** Combine multiple pretrained models for better generalization.

### 1.1 Available Models for Ensemble

| Model | File | Classes | Use Case |
|-------|------|---------|----------|
| Binary | `AntiSpoofing_bin_128.pth` | 2 | Primary spoof detection |
| Print-Replay | `AntiSpoofing_print-replay_128.pth` | 3 | Attack-type aware |
| Print-Replay 1.5 | `AntiSpoofing_print-replay_1.5_128.pth` | 3 | Variant model |

### 1.2 Ensemble Strategy

```
Input Image (128x128)
        │
        ├──────────────────┬──────────────────┐
        ▼                  ▼                  ▼
   FeatherNet_1       FeatherNet_2       FeatherNet_3
   (Binary Model)     (Print-Replay)     (Print-Replay 1.5)
        │                  │                  │
        ▼                  ▼                  ▼
   Spoof Prob 1       Spoof Prob 2       Spoof Prob 3
        │                  │                  │
        └──────────────────┴──────────────────┘
                           │
                           ▼
                    Fusion Strategy
                    (weighted avg / learned)
                           │
                           ▼
                    Final Prediction
```

### 1.3 Fusion Options

1. **Simple Average** (baseline)
   - Average predictions from all 3 models
   - No training required
   - Quick to implement

2. **Weighted Average** (recommended first)
   - Learn optimal weights per model
   - Weights learned on validation set
   - `final = w1*p1 + w2*p2 + w3*p3` where `sum(w) = 1`

3. **Learned Fusion Head** (if needed)
   - Concatenate features from all models
   - Small MLP on top
   - Train fusion head only (freeze backbones)

### 1.4 Implementation Tasks

- [ ] Create `EnsembleModel` class in `models/ensemble.py`
- [ ] Load all 3 pretrained models
- [ ] Implement prediction averaging
- [ ] Implement weighted ensemble with learnable weights
- [ ] Add feature extraction from all models
- [ ] Create `EnsemblePredictor` wrapper

### 1.5 Files to Create/Modify

| File | Action |
|------|--------|
| `models/ensemble.py` | Create - Ensemble architecture |
| `utils/model_loader.py` | Modify - Add ensemble loading |
| `configs/ensemble_config.yaml` | Create - Ensemble settings |

---

## Step 2: Domain Adaptation

**Goal:** Fix cross-dataset performance (SIW: 20% → 60%+).

### 2.1 Problem Analysis

- Model trained on CelebA-Spoof
- Works okay on OULU (65%) - similar domain
- Fails on SIW (20%) - different capture conditions, lighting, attacks

### 2.2 Domain Adaptation Strategies

#### Strategy A: Combined Dataset Training (Simple)
- Merge OULU + SIW into single training set
- Fine-tune ensemble fusion head
- Improves generalization through data diversity

#### Strategy B: Domain-Adversarial Training (Advanced)
- Add domain classifier head
- Train to confuse domain classifier (gradient reversal)
- Forces domain-invariant features

#### Strategy C: Test-Time Adaptation (No retraining)
- Use entropy minimization at test time
- Adapt batch normalization statistics
- Works without labeled target data

### 2.3 Recommended Approach

**Phase 1:** Combined dataset training (Strategy A)
- Merge OULU + SIW datasets
- Balance classes (undersample majority / oversample minority)
- Train fusion head on combined data

**Phase 2:** If needed, add domain adversarial loss (Strategy B)

### 2.4 Implementation Tasks

- [ ] Create combined dataset loader (OULU + SIW)
- [ ] Implement class balancing (weighted sampling)
- [ ] Add domain labels to datasets
- [ ] Implement domain adaptation loss (optional)
- [ ] Create cross-dataset evaluation protocol

### 2.5 Files to Create/Modify

| File | Action |
|------|--------|
| `utils/datasets.py` | Modify - Add combined loader with balancing |
| `utils/domain_adaptation.py` | Create - DA utilities (if needed) |
| `configs/training_config.yaml` | Modify - Add DA settings |

---

## Step 3: Training Pipeline Integration

**Goal:** Complete training pipeline for ensemble + domain adaptation.

### 3.1 Training Strategy

```
Phase 1: Freeze all backbones
         Train fusion weights only
         Epochs: 10-20
         LR: 1e-3

Phase 2: Unfreeze last backbone layers
         Fine-tune with lower LR
         Epochs: 10-20
         LR: 1e-5 (backbone), 1e-4 (fusion)
```

### 3.2 Loss Function

```python
total_loss = classification_loss + lambda_domain * domain_loss

# Classification: Binary Cross-Entropy with label smoothing
# Domain: Cross-entropy on domain labels (with gradient reversal)
```

### 3.3 Implementation Tasks

- [ ] Complete `train.py` with ensemble support
- [ ] Add learning rate scheduling (cosine annealing)
- [ ] Implement gradient accumulation (for larger effective batch)
- [ ] Add mixed precision training (if CUDA available)
- [ ] Implement early stopping
- [ ] Add checkpoint saving (best + periodic)
- [ ] Integrate TensorBoard logging
- [ ] Add WandB logging (optional)

### 3.4 Files to Create/Modify

| File | Action |
|------|--------|
| `train.py` | Modify - Complete implementation |
| `train_ensemble.py` | Create - Ensemble-specific training |
| `utils/trainer.py` | Create - Training utilities |

---

## Step 4: Evaluation & Testing

**Goal:** Comprehensive evaluation with cross-dataset protocols.

### 4.1 Evaluation Protocols

1. **Intra-dataset** (train and test on same dataset)
   - OULU: Train on OULU train, test on OULU test
   - SIW: Train on SIW train, test on SIW test

2. **Cross-dataset** (train on one, test on another)
   - Train: OULU → Test: SIW
   - Train: SIW → Test: OULU
   - Train: OULU+SIW → Test: held-out from both

3. **Leave-one-out**
   - Train on OULU, test on SIW
   - Train on SIW, test on OULU

### 4.2 Metrics to Report

| Metric | Description | Target |
|--------|-------------|--------|
| Accuracy | Overall correct predictions | >80% |
| APCER | Attack (spoof) classification error | <15% |
| BPCER | Bona fide (real) classification error | <15% |
| ACER | Average of APCER and BPCER | <15% |
| AUC | Area under ROC curve | >0.85 |
| EER | Equal Error Rate | <15% |

### 4.3 Implementation Tasks

- [ ] Complete `evaluate.py` with all protocols
- [ ] Add per-attack-type analysis
- [ ] Generate confusion matrices
- [ ] Create ROC curve plots
- [ ] Add threshold optimization
- [ ] Generate evaluation reports (markdown/JSON)

### 4.4 Files to Create/Modify

| File | Action |
|------|--------|
| `evaluate.py` | Modify - Complete implementation |
| `evaluate_ensemble.py` | Create - Ensemble evaluation |
| `utils/evaluation.py` | Create - Evaluation utilities |

---

## Step 5: Inference API Refinement

**Goal:** Production-ready inference with ensemble model.

### 5.1 API Features

- Single image inference
- Batch inference
- Video inference with temporal smoothing
- Webcam real-time inference
- Confidence calibration
- Threshold configuration

### 5.2 Implementation Tasks

- [ ] Update `FASPredictor` for ensemble
- [ ] Add temporal smoothing for video
- [ ] Implement confidence calibration
- [ ] Add input validation
- [ ] Optimize for real-time (batch frames)
- [ ] Add GPU memory management

### 5.3 Files to Create/Modify

| File | Action |
|------|--------|
| `inference.py` | Modify - Add ensemble support |
| `utils/model_loader.py` | Modify - Ensemble predictor |

---

## Step 6: Model Export (Optional)

**Goal:** Export for deployment if needed.

### 6.1 Export Formats

- [ ] ONNX export
- [ ] TorchScript (JIT)
- [ ] CoreML (for iOS)

### 6.2 Implementation Tasks

- [ ] Create export script
- [ ] Validate exported models
- [ ] Benchmark inference speed

### 6.3 Files to Create

| File | Action |
|------|--------|
| `export_model.py` | Create - Export utilities |

---

## Step 7: Streamlit Frontend UI

**Goal:** Basic web UI with webcam live detection.

### 7.1 Core Features (MVP)

- [ ] **Webcam Live Detection**
  - Real-time face detection
  - Spoof/Real classification overlay
  - Confidence score display
  - FPS counter

### 7.2 Future Features (TBD)

- [ ] Image upload and classification
- [ ] Video file processing
- [ ] Attention map visualization
- [ ] Model selection dropdown
- [ ] Batch processing
- [ ] Results history/export

### 7.3 UI Layout (MVP)

```
┌─────────────────────────────────────────────────────┐
│                 FAS - Face Anti-Spoofing            │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │                                             │   │
│  │                                             │   │
│  │              WEBCAM FEED                    │   │
│  │           (with overlay)                    │   │
│  │                                             │   │
│  │                                             │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  [Start Camera]  [Stop Camera]                      │
│                                                     │
│  Status: REAL / SPOOF                               │
│  Confidence: 95.2%                                  │
│  FPS: 24                                            │
│                                                     │
│  ─────────────────────────────────────────────────  │
│  Settings:                                          │
│  Threshold: [────●────] 0.5                         │
│  Model: [Ensemble v]                                │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 7.4 Technical Implementation

```python
# app.py structure
import streamlit as st
import cv2
from utils.model_loader import EnsemblePredictor

# Initialize model (cached)
@st.cache_resource
def load_model():
    return EnsemblePredictor(...)

# Main app
def main():
    st.title("FAS - Face Anti-Spoofing")
    
    # Webcam capture
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])
    
    # Status display
    status_placeholder = st.empty()
    
    # Settings sidebar
    threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.5)
    
    # Main loop
    camera = cv2.VideoCapture(0)
    while run:
        ret, frame = camera.read()
        if ret:
            # Run inference
            result = model.predict(frame)
            
            # Draw overlay
            frame = draw_overlay(frame, result)
            
            # Display
            FRAME_WINDOW.image(frame, channels="BGR")
            status_placeholder.markdown(f"**{result['label']}** ({result['confidence']:.1%})")
    
    camera.release()
```

### 7.5 Dependencies

- `streamlit` - Web framework
- `opencv-python` - Webcam capture
- `streamlit-webrtc` - Better webcam handling (optional)

### 7.6 Implementation Tasks

- [ ] Set up basic Streamlit layout
- [ ] Implement webcam capture with OpenCV
- [ ] Integrate ensemble model
- [ ] Add prediction overlay on video
- [ ] Display status and confidence
- [ ] Add threshold slider
- [ ] Add FPS counter
- [ ] Handle camera errors gracefully
- [ ] Add start/stop controls

### 7.7 Files to Create/Modify

| File | Action |
|------|--------|
| `app.py` | Modify - Complete webcam UI |
| `utils/visualization.py` | Modify - Add overlay drawing |

---

## Implementation Order

```
Step 1 (Ensemble) ──► Step 3 (Training) ──► Step 4 (Evaluation)
                            │
Step 2 (Domain) ────────────┘
                            
Step 5 (Inference) ──► Step 7 (Streamlit UI)
                            │
Step 6 (Export) ────────────┘ (optional)
```

**Recommended Sequence:**

1. **Step 1**: Build ensemble (can test immediately with existing weights)
2. **Step 5**: Refine inference API (needed for UI)
3. **Step 7**: Basic Streamlit UI with webcam
4. **Step 2**: Domain adaptation (improve accuracy)
5. **Step 3**: Training pipeline (train fusion)
6. **Step 4**: Full evaluation
7. **Step 6**: Export (if needed)

---

## Success Criteria

| Milestone | Criteria |
|-----------|----------|
| Ensemble Working | All 3 models loaded, predictions combined |
| Cross-dataset Fixed | SIW accuracy >60% (up from 20%) |
| Training Complete | Can train fusion head, loss decreasing |
| UI MVP | Webcam detection working in browser |
| Production Ready | <100ms inference, stable UI, good accuracy |

---

## Notes

- All training will be on MPS (Apple Silicon) - no mixed precision
- Focus on RGB only (no depth/IR data available)
- Streamlit chosen for simplicity - can migrate to FastAPI later if needed
