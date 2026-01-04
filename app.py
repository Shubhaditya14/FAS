"""Streamlit web application for FAS system."""

import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image

from models.backbones import get_backbone


@st.cache_resource
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_model(model_config_path: str, checkpoint_path: str, device_type: str = "cpu"):
    """Load model from checkpoint."""
    model_config = load_config(model_config_path)

    # Setup device
    if device_type == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_type == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create model
    model = get_backbone(
        name=model_config["backbone"]["name"],
        pretrained=False,
        num_classes=model_config["classifier"]["num_classes"],
    )

    # Load checkpoint
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    return model, device


def get_transform(image_size=(224, 224)) -> A.Compose:
    """Get image transform pipeline."""
    return A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def predict(
    image: np.ndarray, model: nn.Module, transform: A.Compose, device: torch.device
) -> dict:
    """Make prediction on image."""
    # Apply transforms
    transformed = transform(image=image)
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    # Predict
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    inference_time = time.time() - start_time

    is_real = pred_class == 0
    label = "Real" if is_real else "Spoof"

    return {
        "label": label,
        "confidence": confidence,
        "is_real": is_real,
        "probabilities": {"real": probs[0, 0].item(), "spoof": probs[0, 1].item()},
        "inference_time": inference_time,
    }


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Face Anti-Spoofing System", page_icon="🔒", layout="wide"
    )

    # Title and description
    st.title("🔒 Face Anti-Spoofing System")
    st.markdown("""
    This application detects face presentation attacks (spoofing) using deep learning.
    Upload an image or use your webcam to verify if a face is real or a spoof attempt.
    """)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    model_config_path = st.sidebar.text_input(
        "Model Config Path", value="configs/model_config.yaml"
    )

    checkpoint_path = st.sidebar.text_input(
        "Checkpoint Path", value="checkpoints/best_model.pth"
    )

    device_type = st.sidebar.selectbox(
        "Device", options=["cpu", "mps", "cuda"], index=0
    )

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )

    # Load model
    try:
        with st.spinner("Loading model..."):
            model, device = load_model(model_config_path, checkpoint_path, device_type)
            transform = get_transform()
        st.sidebar.success(f"✅ Model loaded on {device}")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        st.stop()

    # Main interface
    st.header("📸 Input")

    input_method = st.radio(
        "Select input method:",
        options=["Upload Image", "Use Webcam", "Sample Images"],
        horizontal=True,
    )

    image = None

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png", "bmp"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image = np.array(image)

    elif input_method == "Use Webcam":
        camera_image = st.camera_input("Take a picture")

        if camera_image is not None:
            image = Image.open(camera_image).convert("RGB")
            image = np.array(image)

    elif input_method == "Sample Images":
        st.info(
            "Sample images feature - add your sample images to 'data/samples/' directory"
        )
        sample_dir = Path("data/samples")
        if sample_dir.exists():
            sample_files = list(sample_dir.glob("*.jpg")) + list(
                sample_dir.glob("*.png")
            )
            if sample_files:
                selected_sample = st.selectbox(
                    "Choose a sample:", options=[f.name for f in sample_files]
                )
                if selected_sample:
                    image = Image.open(sample_dir / selected_sample).convert("RGB")
                    image = np.array(image)

    # Prediction
    if image is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Input Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("🔍 Analysis Results")

            with st.spinner("Analyzing..."):
                result = predict(image, model, transform, device)

            # Display results
            if result["is_real"]:
                st.success(f"✅ **{result['label']}**")
                status_color = "green"
            else:
                st.error(f"⚠️ **{result['label']}**")
                status_color = "red"

            # Confidence meter
            st.metric(label="Confidence", value=f"{result['confidence']:.1%}")

            # Probability breakdown
            st.subheader("📊 Probability Distribution")
            prob_col1, prob_col2 = st.columns(2)

            with prob_col1:
                st.metric(label="Real", value=f"{result['probabilities']['real']:.1%}")

            with prob_col2:
                st.metric(
                    label="Spoof", value=f"{result['probabilities']['spoof']:.1%}"
                )

            # Progress bars
            st.progress(result["probabilities"]["real"], text="Real probability")
            st.progress(result["probabilities"]["spoof"], text="Spoof probability")

            # Additional info
            st.divider()
            st.caption(f"⏱️ Inference time: {result['inference_time'] * 1000:.1f}ms")
            st.caption(f"🖥️ Device: {device}")

            # Decision based on threshold
            st.divider()
            st.subheader("🎯 Decision")
            if result["confidence"] >= confidence_threshold:
                if result["is_real"]:
                    st.success(
                        f"✅ **ACCEPT** - Real face detected with high confidence"
                    )
                else:
                    st.error(
                        f"❌ **REJECT** - Spoof attack detected with high confidence"
                    )
            else:
                st.warning(
                    f"⚠️ **UNCERTAIN** - Confidence below threshold ({confidence_threshold:.0%})"
                )

    # Information section
    st.divider()
    st.header("ℹ️ About")

    with st.expander("How it works"):
        st.markdown("""
        This Face Anti-Spoofing (FAS) system uses deep learning to detect presentation attacks:

        1. **Input Processing**: The image is preprocessed and normalized
        2. **Feature Extraction**: A deep neural network extracts facial features
        3. **Classification**: The model classifies the face as Real or Spoof
        4. **Confidence Score**: Provides probability estimates for both classes

        **Common Spoof Types Detected:**
        - 📄 Printed photos
        - 📱 Screen replays (phone/tablet)
        - 🎭 Masks
        - 📹 Video replays
        """)

    with st.expander("Performance Metrics"):
        st.markdown("""
        Key metrics for face anti-spoofing:

        - **APCER**: Attack Presentation Classification Error Rate
        - **BPCER**: Bona Fide Presentation Classification Error Rate
        - **ACER**: Average Classification Error Rate
        - **EER**: Equal Error Rate
        """)

    with st.expander("Tips for Best Results"):
        st.markdown("""
        - ✅ Use good lighting conditions
        - ✅ Ensure the face is clearly visible
        - ✅ Avoid extreme angles or occlusions
        - ✅ Use high-resolution images when possible
        - ❌ Avoid heavily compressed or blurry images
        """)


if __name__ == "__main__":
if __name__ == '__main__':
    main()
