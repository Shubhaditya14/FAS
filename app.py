"""Streamlit web application for FAS system."""

import time
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from models.feathernet import create_feathernet


@st.cache_resource
def load_model(checkpoint_path: str, device: torch.device):
    """Load FeatherNet model from checkpoint."""
    model = create_feathernet(num_classes=2, pretrained_path=checkpoint_path, device=str(device))
    model.eval()
    return model


def get_transform(image_size: tuple = (128, 128)) -> T.Compose:
    """Get image transform pipeline."""
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def predict(
    image: Image.Image,
    model: nn.Module,
    transform: T.Compose,
    device: torch.device
) -> dict:
    """Make prediction on image."""
    # Apply transforms
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Predict
    start_time = time.time()
    with torch.no_grad():
        spoof_prob = model(image_tensor)
        confidence = spoof_prob.item()
        is_real = confidence < 0.5
        label = "Real" if is_real else "Spoof"
        real_prob = 1.0 - confidence
    inference_time = time.time() - start_time

    return {
        "label": label,
        "confidence": confidence if not is_real else real_prob,
        "is_real": is_real,
        "probabilities": {"real": real_prob, "spoof": confidence},
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

    checkpoint_dir = Path("pth")
    if checkpoint_dir.exists():
        available_models = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pth.tar"))
    else:
        available_models = []

    if not available_models:
        st.sidebar.error("❌ No checkpoint files found in pth/ directory")
        st.stop()

    model_names = [str(p) for p in available_models]
    checkpoint_path = st.sidebar.selectbox("Select Model", model_names)

    device_type = st.sidebar.selectbox(
        "Device", options=["cpu", "mps", "cuda"], index=1 if torch.backends.mps.is_available() else 0
    )
    device = torch.device(device_type)

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )

    # Load model
    try:
        with st.spinner("Loading model..."):
            model = load_model(checkpoint_path, device)
            transform = get_transform()
            num_params = sum(p.numel() for p in model.parameters())
        st.sidebar.success(f"✅ Model loaded on {device}")
        st.sidebar.info(f"📊 Parameters: {num_params:,}")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        st.stop()

    # Main interface
    st.header("📸 Input")

    input_method = st.radio(
        "Select input method:",
        options=["Upload Image", "Use Webcam"],
        horizontal=True,
    )

    image = None

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png", "bmp"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

    elif input_method == "Use Webcam":
        camera_image = st.camera_input("Take a picture")

        if camera_image is not None:
            image = Image.open(camera_image).convert("RGB")

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
        2. **Feature Extraction**: FeatherNet extracts facial features
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
    main()
