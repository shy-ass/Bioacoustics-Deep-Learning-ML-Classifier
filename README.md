# Bioacoustics-Deep-Learning-ML-Classifier
This repository contains a full lifecycle machine learning and deep learning pipeline designed to classify biological acoustic events across 15 animal species. The architecture specifically addresses extreme class imbalances and is explicitly engineered for local hardware execution utilizing CUDA and Automatic Mixed Precision.
Hardware and Environment Setup
This pipeline was developed and optimized for the following environment:

Environment: Conda
Runtime: Python 3.13.11

# Phase 1: Dataset Acquisition
The dataset consists of 908 audio files categorized into 15 distinct animal classes. Raw audio files are stored in the dataset/ directory. Due to the biological rarity of certain species (such as wolves), the dataset contains severe class imbalances, which necessitated strict stratified splitting and algorithmic compensation in later phases.

# Phase 2: Classical Machine Learning (ML)
Located in the ml/ directory, this phase approaches the problem through tabular data and statistical modeling.
Feature Extraction: Time-series audio arrays are mathematically flattened into 526 distinct spectral and temporal features (MFCCs, spectral contrast, zero-crossing rates, etc.).
Dimensionality Reduction: A Recursive Feature Elimination with Cross-Validation (RFECV) pipeline filters out acoustic noise, reducing the 526 features to the 26 most critical mathematical signals.
Model Training: The refined features are used to train and evaluate a jury of models, including LightGBM, XGBoost, Random Forest, and SVM. The compiled artifacts are saved for rapid, CPU-based inference.

#Phase 3: Deep Learning (DL)
Located in the dl/ directory, this phase abandons flat statistical averages and treats acoustic classification as a computer vision problem to capture the temporal shape of soundwaves.
Energy Tracking: A custom PyTorch Dataset class scans audio files to extract the loudest 5-second acoustic event, preventing arbitrary cropping of biological calls.
Spectrogram Conversion: 1D audio arrays are converted into 3-channel RGB Mel-Spectrogram tensors natively on the GPU.
Model Training: The pipeline utilizes transfer learning on pre-trained vision architectures (EfficientNet-B0, ResNet34, MobileNetV3) alongside a custom 1D RawAudio CNN.
Hardware Optimization: To prevent VRAM overflow on the 6 GB RTX 4050, the training loop implements dynamic batch sizing, Automatic Mixed Precision (AMP), and aggressive garbage collection between model initializations.

# Phase 4: Local Web Deployment
The backend pipelines are unified under a single Streamlit web dashboard for user interaction.
The web application caches the heavy deep learning models into VRAM exactly once upon initialization to maintain system stability.
Users can select random files from the local dataset or upload new audio files.
Both the ML and DL models process the audio simultaneously to provide a comparative inference report.

# Execution Instructions
To run this application locally on your machine:
Clone this repository.
Install the exact library dependencies listed in requirements.txt.
Open your terminal, navigate to the root directory of the project, and execute the following command:

python run.py

This will automatically launch the local Streamlit server and open the dashboard in your default web browser.
