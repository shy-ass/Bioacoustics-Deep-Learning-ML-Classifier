import streamlit as st
import os
import random
import tempfile
from pathlib import Path

# Set the page configuration to wide mode
st.set_page_config(page_title="Bioacoustics Classifier", page_icon="🐾", layout="wide")

# ==============================================================================
# 1. ARTIFACT CACHING (GPU MEMORY MANAGEMENT)
# ==============================================================================
# The @st.cache_resource decorator ensures your RTX 4050 only loads these 
# heavy models into VRAM exactly once when the server boots up.

@st.cache_resource
def load_ml_jury():
    # Placeholder: return load_ml_models()
    return "ML Models Loaded"

@st.cache_resource
def load_dl_jury():
    # Placeholder: return load_dl_models()
    return "DL Models Loaded"

ml_models = load_ml_jury()
dl_models = load_dl_jury()

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
DATASET_PATH = "dataset" # Assumes app.py is in ~/audio and dataset is ~/audio/dataset

def get_animal_classes(dataset_path=DATASET_PATH):
    """Scans the dataset directory and returns just the clean folder names."""
    if not os.path.exists(dataset_path):
        return []
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    return sorted(classes)

def pick_random_file_from_class(dataset_path, animal_class):
    """Silently grabs a random audio file from the selected class folder."""
    class_dir = os.path.join(dataset_path, animal_class)
    valid_exts = {'.wav', '.mp3', '.ogg', '.flac'}
    
    if not os.path.exists(class_dir):
        return None
        
    files = [f for f in os.listdir(class_dir) if Path(f).suffix.lower() in valid_exts]
    if not files:
        return None
        
    random_file = random.choice(files)
    return os.path.join(class_dir, random_file)

# ==============================================================================
# 3. UI LAYOUT
# ==============================================================================

st.title("🐾 Bioacoustics Deep Learning & ML Classifier")
st.markdown("Upload an animal sound or pick a random animal from the dataset, and our local AI Jury will deliberate on what is making the noise.")

st.markdown("---")

# Input Section
col_input1, col_input2 = st.columns(2)

with col_input1:
    st.subheader("Option 1: Pick a Random Animal")
    animal_classes = get_animal_classes()
    
    selected_class = st.selectbox(
        "Select an animal to test:", 
        options=["-- Select an animal --"] + animal_classes
    )

with col_input2:
    st.subheader("Option 2: Drag and Drop")
    uploaded_file = st.file_uploader("Upload an audio file (.wav, .mp3)", type=["wav", "mp3", "ogg"])

st.markdown("---")

# ==============================================================================
# 4. EXECUTION LOGIC
# ==============================================================================

target_file_path = None
display_name = ""

# Handle User Inputs
if uploaded_file is not None:
    # Save the uploaded file to a temporary location so the local GPU can process it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        target_file_path = tmp_file.name
        display_name = f"Uploaded File: {uploaded_file.name}"
        st.success("File uploaded successfully into local memory.")
        
elif selected_class != "-- Select an animal --":
    # Silently grab a random file from the chosen folder
    target_file_path = pick_random_file_from_class(DATASET_PATH, selected_class)
    if target_file_path:
        # Hide the messy file path from the user, just show them it worked
        display_name = f"Random '{selected_class}' sample selected from local dataset."
        st.success(display_name)
    else:
        st.error(f"Could not find any audio files in the {selected_class} folder.")

# Run the Juries if a file is securely staged
if target_file_path:
    st.audio(target_file_path) # Let the user hear what the GPU is about to process
    
    if st.button("🚀 Run Local AI Analysis", use_container_width=True):
        
        with st.spinner("Local neural networks are computing..."):
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.header("📊 Classical ML Jury")
                st.caption("XGBoost, LightGBM, Random Forest, SVM")
                
                # --- HOOK UP YOUR ML INFERENCE HERE ---
                st.info("Predicted: **Dog** (Confidence: 68%)") # Placeholder
                with st.expander("View individual model votes"):
                    st.write("- LightGBM: Dog")
                    st.write("- XGBoost: Dog")
            
            with res_col2:
                st.header("👁️ Deep Learning Jury")
                st.caption("ResNet34, EfficientNet-B0, MobileNetV3, RawAudioCNN")
                
                # --- HOOK UP YOUR DL INFERENCE HERE ---
                st.success("Predicted: **Wolf (Howl)** (Confidence: 91%)") # Placeholder
                with st.expander("View individual model votes"):
                    st.write("- EfficientNet_B0: Wolf (Howl)")
                    st.write("- ResNet34: Wolf (Howl)")

    # Clean up the temporary file to keep the SSD clear
    if uploaded_file is not None and os.path.exists(target_file_path):
        os.remove(target_file_path)
