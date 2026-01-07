"""Streamlit app for multi-model ensemble FAS."""

from pathlib import Path

import streamlit as st
from PIL import Image

from multi_model_predictor import MultiModelPredictor, create_simple_ensemble, create_weighted_ensemble


@st.cache_resource
def load_ensemble(
    checkpoint_paths: list,
    device: str,
    fusion_type: str = "average",
    weights: list = None,
):
    """Load ensemble model."""
    if fusion_type == "weighted" and weights is None:
        weights = [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)

    return MultiModelPredictor(
        checkpoint_paths=checkpoint_paths,
        device=device,
        fusion_type=fusion_type,
        weights=weights,
    )
    """Load ensemble model."""
    return MultiModelPredictor(
        checkpoint_paths=checkpoint_paths,
        device=device,
        fusion_type=fusion_type,
        weights=weights,
    )


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Ensemble Face Anti-Spoofing",
        page_icon="👤",
        layout="wide",
    )

    st.title("👤 Ensemble Face Anti-Spoofing Detection")
    st.markdown("---")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Device selection
    device_option = st.sidebar.radio(
        "Device",
        ["cpu", "mps", "cuda"],
        index=1 if torch.backends.mps.is_available() else 0,
    )
    device = device_option

    # Checkpoint directory
    checkpoint_dir = Path("pth")
    if checkpoint_dir.exists():
        available_models = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pth.tar"))
    else:
        available_models = []

    if not available_models:
        st.error("No checkpoint files found in pth/ directory")
        return

    # Model selection
    st.sidebar.subheader("📁 Model Selection")
    select_all = st.sidebar.checkbox("Select all models", value=True)

    if select_all:
        selected_models = [str(p) for p in available_models]
    else:
        selected_models = st.sidebar.multiselect(
            "Select models",
            options=[str(p) for p in available_models],
            default=[str(p) for p in available_models],
        )

    if not selected_models:
        st.warning("Please select at least one model")
        return

    # Fusion strategy
    st.sidebar.subheader("🔀 Fusion Strategy")
    fusion_type = st.sidebar.selectbox(
        "Fusion method",
        options=["average", "weighted", "max", "voting"],
        index=0,
    )

    weights = None
    if fusion_type == "weighted":
        st.sidebar.info("Weighted fusion: Adjust weights for each model")
        num_models = len(selected_models)
        default_weights = [1.0 / num_models] * num_models

        weights_input = st.sidebar.text_input(
            "Enter weights (comma-separated, must sum to 1.0)",
            value=",".join([f"{w:.2f}" for w in default_weights]),
        )

        try:
            weights = [float(w.strip()) for w in weights_input.split(",")]
            if len(weights) != num_models:
                st.error(f"Expected {num_models} weights, got {len(weights)}")
                weights = None
            elif abs(sum(weights) - 1.0) > 1e-6:
                st.error(f"Weights must sum to 1.0, got {sum(weights):.4f}")
                weights = None
        except ValueError:
            st.error("Invalid weights format. Use comma-separated numbers.")
            weights = None

    # Load ensemble
    with st.sidebar:
        with st.spinner("Loading ensemble..."):
            try:
                ensemble = load_ensemble(
                    checkpoint_paths=selected_models,
                    device=device,
                    fusion_type=fusion_type,
                    weights=weights,
                )
                st.success(f"✅ Ensemble loaded ({len(selected_models)} models)")
                st.info(f"🔀 Fusion: {fusion_type}")
            except Exception as e:
                st.error(f"❌ Error loading ensemble: {e}")
                return

    # Main interface
    tab1, tab2 = st.tabs(["📷 Image Upload", "📹 Camera"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
            help="Upload a face image for spoof detection",
        )

        if uploaded_file is not None:
            process_image(uploaded_file, ensemble, selected_models)

    with tab2:
        camera_image = st.camera_input("Capture face image")

        if camera_image is not None:
            process_image(camera_image, ensemble, selected_models)

    # Information section
    st.markdown("---")
    st.subheader("ℹ️ About")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **How Ensemble Works:**

        This system uses multiple pre-trained models and combines their predictions
        using different fusion strategies:

        - **Average**: Simple average of all model predictions
        - **Weighted**: Weighted average with custom weights
        - **Max**: Maximum confidence across all models
        - **Voting**: Majority voting (each model votes)

        Ensembles typically provide better generalization across different
        datasets and attack types.
        """)

    with col2:
        st.markdown("""
        **Available Models:**

        The following pre-trained models are available:

        - **binary**: General binary classifier
        - **print-replay**: Optimized for print and replay attacks
        - **print-replay 1.5**: Enhanced version of print-replay

        You can select any combination of models and adjust fusion
        weights to optimize for your specific use case.
        """)


def process_image(image_file, ensemble, model_names):
    """Process uploaded or captured image."""
    col1, col2 = st.columns([1, 1])

    with col1:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Input Image", use_column_width=True)

    with col2:
        with st.spinner("Analyzing with ensemble..."):
            result = ensemble.predict(image, return_individual=True)

        # Display ensemble result
        st.subheader("🎯 Ensemble Result")

        if result["is_real"]:
            st.success(f"✅ **{result['label']}**")
            confidence_color = "green"
        else:
            st.error(f"⚠️ **{result['label']}**")
            confidence_color = "red"

        st.metric(
            label="Confidence",
            value=f"{result['confidence']:.2%}",
        )

        # Progress bars
        col_a, col_b = st.columns(2)
        with col_a:
            st.progress(result["probabilities"]["real"], text="Real probability")
        with col_b:
            st.progress(result["probabilities"]["spoof"], text="Spoof probability")

        # Individual model predictions
        if "individual_predictions" in result:
            st.subheader("📊 Individual Model Predictions")

            for model_name, prob in result["individual_predictions"].items():
                with st.expander(f"📁 {model_name}"):
                    is_real_model = prob < 0.5
                    model_label = "Real" if is_real_model else "Spoof"
                    model_confidence = prob if not is_real_model else (1.0 - prob)

                    if is_real_model:
                        st.success(f"**{model_label}** - {model_confidence:.2%}")
                    else:
                        st.error(f"**{model_label}** - {model_confidence:.2%}")

                    col_x, col_y = st.columns(2)
                    with col_x:
                        st.progress(1.0 - prob, text=f"Real: {1.0-prob:.2%}")
                    with col_y:
                        st.progress(prob, text=f"Spoof: {prob:.2%}")


if __name__ == "__main__":
    import torch
    main()
