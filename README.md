# Identification of Shipping Signals with Few-Shot Learning: A Distribution-Aware Approach

Official implementation of the paper **"Identification of Shipping Signals with Few-Shot Learning: A Distribution-Aware Approach"** submitted to **PLOS ONE**.

This repository provides a few-shot learning (FSL) framework specifically designed for underwater acoustic signal identification. By integrating **Hard Negative Softmax (HNS)** and **Boundary Loss (BL)**, the model constructs a discriminative embedding space that effectively handles the high variability and data scarcity inherent in maritime environments.

## 1. Code Overview
The project is organized into three main modules, optimized for single-GPU execution and reproducibility:

* **`analysis_utils.py` (Core Logic & Data Handling)**:
    * Implements the proposed **Boundary Negative Contrastive Loss**.
    * Handles N-way N-shot episode sampling and data normalization (0–1).
    * Includes a stochastic Gaussian noise injection strategy (std=0.02) to simulate real-world underwater ambient noise.
* **`model_deep.py` (Architecture)**:
    * Defines the Siamese network using a **VGG-16 backbone** for 512-dimensional latent embeddings.
    * Contains the core training step logic for distribution-aware feature extraction.
* **`run_experiment.py` (Main Execution)**:
    * **Single GPU Optimization**: Specifically configured to run efficiently on a single NVIDIA GPU with memory growth settings.
    * **Hyperparameter Setup**: Pre-configured with optimal settings (**700 Epochs**, Learning Rate 2e-5, and Boundary Weight 0.3).
    * Automates the end-to-end workflow: dataset loading, training, and performance logging.

## 2. Environment & Requirements
* **Language**: Python 3.9–3.10
* **Key Packages**:
    * `tensorflow >= 2.10`
    * `numpy`, `pandas`, `matplotlib`
* **Hardware**: A single NVIDIA GPU with CUDA/CuDNN is strongly recommended.

```bash
pip install "tensorflow>=2.10" numpy pandas matplotlib
3. Data Preparation
The dataset consists of shipping signals (Cargo, Tanker, Container, etc.) collected in the southern sea of Jeju Island.

Preprocessing: Acoustic signals are converted into (224×224×3) spectrograms, normalized, and stored as .npy files.

Directory Structure:

Plaintext
./data/ship_type/
├── Class_A/
│   ├── sample1.npy
│   └── ...
└── Class_B/
    └── ...
Path Configuration: Update the folderpath_npy variable in run_experiment.py to match your local dataset path.

4. How to Run
Ensure your preprocessed .npy data is placed in the designated folder.

Run the main training script:

Bash
python run_experiment.py
Outputs:

Real-time training logs per epoch.

Final evaluation metrics, including mean accuracy and standard deviation.

5. Experimental Results
The proposed framework achieves the following performance on the maritime dataset:

Mean Accuracy: 87.81% (in a 5-way 5-shot setup)

Stability: 5.81% Standard Deviation (Lowest among SOTA backbones), ensuring reliable performance in volatile maritime environments.

Convergence: The model is optimized over 700 epochs to ensure a stable and discriminative manifold.

6. Data Availability
Supporting Information: This repository includes two raw acoustic samples and sample spectrograms for demonstration purposes.

Full Dataset: Due to security constraints and file size limitations, the complete raw and preprocessed dataset is available from the corresponding author upon reasonable request.

7. License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact: For any questions regarding the code or research, please contact the corresponding author as listed in the manuscript.
