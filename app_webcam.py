"""Streamlit app for continuous live FAS detection with OpenCV - FIXED VERSION."""

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
    page_title="Face Anti-Spoofing - Live Detection",
    page_icon="🎥",
    layout="wide",
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


def main():
    """Main Streamlit app."""
    
    # Title
    st.title("🎥 Face Anti-Spoofing - Live Detection")
    st.markdown("Real-time face liveness detection - continuous video feed")
    st.markdown("---")
    
    # Initialize session state
    if "detection_running" not in st.session_state:
        st.session_state.detection_running = False
    
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False
    
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
            index=len(device_options) - 1,
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
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Video Feed")
        
        # Video placeholder (will be updated continuously)
        video_placeholder = st.empty()
        
        # Control buttons
        button_col1, button_col2, button_col3 = st.columns(3)
        
        with button_col1:
            start_button = st.button("▶️ Start Detection", type="primary", use_container_width=True)
        
        with button_col2:
            stop_button = st.button("⏹️ Stop Detection", type="secondary", use_container_width=True)
        
        with button_col3:
            if st.session_state.detection_running:
                st.warning("⚠️ Detection Running")
            else:
                st.info("ℹ️ Detection Stopped")
    
    with col2:
        st.subheader("📊 Live Statistics")
        
        # Metrics placeholders
        status_placeholder = st.empty()
        confidence_placeholder = st.empty()
        fps_placeholder = st.empty()
        
        # Show initial state
        if not st.session_state.detection_running:
            status_placeholder.info("Waiting to start detection...")
            video_placeholder.info("📷 Click 'Start Detection' to begin")
        else:
            status_placeholder.info("Detection running...")
            video_placeholder.info("📹 Initializing camera...")
    
    # Detection loop logic
    def run_detection():
        """Run the detection loop."""
        
        try:
            # Open camera
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                st.error(f"❌ Could not open camera {camera_index}")
                st.error("Possible solutions:")
                st.error("1. Check if camera is in use by another app")
                st.error("2. Try different camera index in settings")
                st.error("3. Check browser camera permissions")
                st.session_state.detection_running = False
                return
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            video_placeholder.success("✅ Camera connected - detecting faces...")
            
            fps_counter = 0
            fps_start_time = time.time()
            current_fps = 0
            frame_count = 0
            
            # Process frames
            while st.session_state.detection_running and not st.session_state.stop_requested:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # Update FPS every 10 frames
                fps_counter += 1
                if fps_counter % 10 == 0:
                    elapsed = time.time() - fps_start_time
                    current_fps = fps_counter / elapsed if elapsed > 0 else 0
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # Run inference
                result = inference.predict_image(pil_image, threshold=threshold)
                
                # Draw overlay on frame
                h, w = frame.shape[:2]
                
                # Color based on prediction
                color = (0, 255, 0) if result["is_real"] else (0, 0, 255)  # BGR
                
                # Draw semi-transparent overlay bar
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, -1)
                
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
                
                # Update video display
                video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                
                # Update metrics
                if result["is_real"]:
                    status_placeholder.success(f"✅ **{result['label']}**")
                else:
                    status_placeholder.error(f"⚠️ **{result['label']}**")
                
                confidence_placeholder.metric(
                    "Confidence",
                    f"{result['confidence']:.1%}",
                )
                
                fps_placeholder.metric("FPS", f"{current_fps:.1f}")
            
            # Cleanup
            cap.release()
            st.session_state.detection_running = False
            st.session_state.stop_requested = False
            video_placeholder.info("📷 Detection stopped - camera released")
            
        except Exception as e:
            st.error(f"❌ Detection error: {e}")
            st.session_state.detection_running = False
            video_placeholder.error(f"❌ Camera error occurred")
    
    # Button handlers
    if start_button and not st.session_state.detection_running:
        st.session_state.detection_running = True
        st.session_state.stop_requested = False
        st.rerun()
    
    if stop_button:
        st.session_state.stop_requested = True
        # Don't re-run, just let the loop stop naturally
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About:** This system uses FeatherNet deep learning model for real-time face liveness detection.
    It can detect various spoofing attacks including printed photos, video replays, and masks.
    
    **Tips:**
    - Ensure good lighting for best results
    - Face the camera directly (not angle)
    - Adjust threshold to balance false positives/negatives
    - Green border = Real face detected
    - Red border = Spoof attack detected
    
    **Note:** Continuous video feed updates automatically - no need to capture photos manually.
    """)


if __name__ == "__main__":
    main()
