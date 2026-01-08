"""Streamlit app for FAS with both live video and image upload modes."""

import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from inference import FASInference


# Page configuration
st.set_page_config(
    page_title="Face Anti-Spoofing Detection",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_inference_model(
    checkpoint_path: str,
    device: str,
):
    """Load FASInference model (cached)."""
    model = FASInference(
        checkpoint_paths=checkpoint_path,
        device=device,
        temporal_smoothing=True,
        smoothing_window=5,
    )
    return model


def process_live_frames(
    video_capture: cv2.VideoCapture,
    inference_model,
    threshold: float,
    display_frame_placeholder,
    status_placeholder,
    confidence_placeholder,
    fps_placeholder,
    stop_event,
):
    """Process frames in a loop and update Streamlit UI.
    
    Args:
        video_capture: OpenCV VideoCapture object
        inference_model: FASInference instance
        threshold: Detection threshold
        display_frame_placeholder: Streamlit placeholder for video
        status_placeholder: Placeholder for status
        confidence_placeholder: Placeholder for confidence
        fps_placeholder: Placeholder for FPS
        stop_event: threading.Event to stop processing
    """
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    while not stop_event.is_set():
        ret, frame = video_capture.read()
        
        if not ret:
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Run inference
        result = inference_model.predict_image(pil_image, threshold=threshold)
        
        # Update FPS
        fps_counter += 1
        if fps_counter % 10 == 0:
            elapsed = time.time() - fps_start_time
            current_fps = fps_counter / elapsed if elapsed > 0 else 0
        
        # Draw overlay on frame
        h, w = frame.shape[:2]
        
        # Color based on prediction
        color = (0, 255, 0) if result["is_real"] else (0, 0, 255)  # BGR: Green=Real, Red=Spoof
        
        # Draw semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        alpha = 0.6
        frame = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0.3, -1)
        
        # Draw status text
        label = result["label"]
        confidence = result["confidence"]
        
        # Large status text
        cv2.putText(
            frame,
            f"{label}: {confidence:.1%}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            color,
            3,
            cv2.LINE_AA,
        )
        
        # Draw FPS
        cv2.putText(
            frame,
            f"FPS: {current_fps:.1f}",
            (w - 140, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        
        # Convert BGR to RGB for display
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update Streamlit UI
        display_frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)
        
        if result["is_real"]:
            status_placeholder.success(f"✅ **{result['label']}**")
        else:
            status_placeholder.error(f"⚠️ **{result['label']}**")
        
        confidence_placeholder.metric(
            "Confidence",
            f"{result['confidence']:.1%}",
        )
        
        fps_placeholder.metric("FPS", f"{current_fps:.1f}")
        
        # Small delay to prevent overwhelming
        time.sleep(0.01)


def main():
    """Main Streamlit app."""
    
    # Title
    st.title("🎥 Face Anti-Spoofing Detection")
    st.markdown("Real-time and image-based face liveness detection")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Device selection
        device_options = ["cpu"]
        if torch.backends.mps.is_available():
            device_options.append("mps")
        if torch.cuda.is_available():
            device_options.append("cuda")
        
        device = st.selectbox(
            "Device",
            device_options,
            index=len(device_options) - 1,  # Default to best available
            help="Computation device (MPS/CUDA for faster inference)",
        )
        
        st.markdown("---")
        
        # Model selection
        st.subheader("📦 Model Configuration")
        
        checkpoint_dir = Path("pth")
        available_models = []
        if checkpoint_dir.exists():
            available_models = sorted(
                list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pth.tar"))
            )
        
        if not available_models:
            st.error("❌ No models found in pth/ directory")
            st.stop()
        
        # Filter to binary models
        binary_models = [p for p in available_models if "bin" in p.name.lower()]
        
        if not binary_models:
            st.warning("⚠️ No binary models found, using first available")
            binary_models = available_models[:1]
        
        model_names = [p.name for p in binary_models]
        
        selected_model = st.selectbox(
            "Select Model",
            model_names,
            index=0,
            help="Choose pretrained model",
        )
        
        selected_checkpoint = str(checkpoint_dir / selected_model)
        
        st.markdown("---")
        
        # Inference settings
        st.subheader("🎯 Inference Settings")
        
        threshold = st.slider(
            "Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Threshold for classifying as spoof (higher = stricter)",
        )
        
        camera_index = st.number_input(
            "Camera Index",
            min_value=0,
            max_value=10,
            value=0,
            help="Camera device index (usually 0 for built-in camera)",
        )
        
        st.markdown("---")
        
        # Load model
        with st.spinner("Loading model..."):
            try:
                inference = load_inference_model(
                    checkpoint_path=selected_checkpoint,
                    device=device,
                )
                st.success("✅ Model loaded successfully")
                
                # Model info
                if hasattr(inference.model, 'parameters'):
                    num_params = sum(p.numel() for p in inference.model.parameters())
                    st.metric("Parameters", f"{num_params:,}")
                
            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
                st.stop()
    
    # Mode selection
    mode = st.radio(
        "Detection Mode",
        ["📹 Live Webcam", "📸 Image Upload"],
        horizontal=True,
    )
    
    if mode == "📹 Live Webcam":
        # Live webcam mode
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📹 Live Video Feed")
            
            # Control buttons
            button_col1, button_col2, button_col3 = st.columns(3)
            
            with button_col1:
                start_button = st.button("▶️ Start Detection", type="primary", use_container_width=True)
            
            with button_col2:
                stop_button = st.button("⏹️ Stop", type="secondary", use_container_width=True)
            
            with button_col3:
                if st.session_state.get("detection_running", False):
                    st.warning("⚠️ Detection Running")
            
        with col2:
            st.subheader("📊 Live Statistics")
            
            # Metrics placeholders
            status_placeholder = st.empty()
            confidence_placeholder = st.empty()
            fps_placeholder = st.empty()
            
            # Video placeholder (will be updated continuously)
            video_placeholder = st.empty()
        
        # Instructions
        st.markdown("---")
        st.subheader("📖 How to Use")
        
        col_i1, col_i2, col_i3 = st.columns(3)
        
        with col_i1:
            st.markdown("""
                **1. Configure Settings** (sidebar)
                   - Select device (MPS recommended for Mac)
                   - Choose model (AntiSpoofing_bin_128.pth recommended)
                   - Adjust detection threshold if needed
                """)
        
        with col_i2:
            st.markdown("""
                **2. Start Detection**
                   - Click "▶️ Start Detection" button
                   - Allow camera permission in browser
                   - Position your face in camera frame
                """)
        
        with col_i3:
            st.markdown("""
                **3. Watch Results**
                   - ✅ Green border = Real face
                   - ⚠️ Red border = Spoof attack
                   - Monitor confidence and FPS
                """)
    
    else:
        # Image upload mode
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📸 Upload Image")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png", "bmp"],
                help="Upload a face image for analysis",
            )
            
            # Or camera capture
            st.markdown("**Or capture with camera:**")
            camera_photo = st.camera_input("Capture from camera", key="camera_upload")
            
        with col2:
            st.subheader("📊 Analysis Results")
        
        # Process image
        image_to_analyze = None
        
        if uploaded_file is not None:
            image_to_analyze = Image.open(uploaded_file).convert("RGB")
        elif camera_photo is not None:
            image_to_analyze = Image.open(camera_photo).convert("RGB")
        
        if image_to_analyze is not None:
            with st.spinner("Analyzing..."):
                start_time = time.time()
                result = inference.predict_image(image_to_analyze, threshold=threshold)
                inference_time = time.time() - start_time
            
            # Display image
            st.image(image_to_analyze, caption="Input Image", use_container_width=True)
            
            # Display results
            if result["is_real"]:
                st.success(f"✅ **{result['label']}**")
                status_emoji = "✅"
            else:
                st.error(f"⚠️ **{result['label']}**")
                status_emoji = "⚠️"
            
            # Metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Confidence", f"{result['confidence']:.1%}")
            with col_b:
                st.metric("Inference Time", f"{inference_time*1000:.0f}ms")
            
            # Probability breakdown
            st.markdown("**Probabilities:**")
            st.progress(result['probabilities']['real'], text=f"Real: {result['probabilities']['real']:.1%}")
            st.progress(result['probabilities']['spoof'], text=f"Spoof: {result['probabilities']['spoof']:.1%}")
            
            # Raw output for debugging
            with st.expander("🔍 Debug Info"):
                st.json({
                    "label": result["label"],
                    "is_real": result["is_real"],
                    "confidence": f"{result['confidence']:.4f}",
                    "real_prob": f"{result['probabilities']['real']:.4f}",
                    "spoof_prob": f"{result['probabilities']['spoof']:.4f}",
                    "threshold": threshold,
                })
        else:
            st.info("📷 Upload an image or capture with camera to begin analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About:** This system uses FeatherNet deep learning model for face liveness detection.
    It can detect various spoofing attacks including printed photos, video replays, and masks.
    
    **Detection Modes:**
    - **Live Webcam**: Continuous video feed with real-time classification
    - **Image Upload**: Analyze individual photos
    
    **Tips:**
    - Use good lighting for best results
    - Face the camera directly (not at angle)
    - Adjust threshold to balance false positives/negatives
    - Green border = Real face, Red border = Spoof attack
    """)


if __name__ == "__main__":
    main()
