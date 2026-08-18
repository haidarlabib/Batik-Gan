"""
Module: config.py
Deskripsi: Konfigurasi global dan parameter inferensi untuk Batik AI Generator (DCGAN).
"""

import os

# Konfigurasi Model DCGAN
IMAGE_SIZE = 64
NZ = 100               # Dimensi vektor laten
NGF = 64              # Ukuran filter dasar Generator
NC = 3                # Channel output (RGB)

# Candidate Checkpoint Paths (Relative Paths)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CANDIDATE_MODEL_PATHS = [
    os.path.join(BASE_DIR, "models", "generator_final.pth"),
    os.path.join(BASE_DIR, "outputs", "checkpoints", "generator_final.pth"),
    os.path.join(BASE_DIR, "outputs", "checkpoints", "generator_epoch_015.pth"),
    os.path.join(BASE_DIR, "outputs", "checkpoints", "generator_epoch_010.pth"),
    os.path.join(BASE_DIR, "outputs", "checkpoints", "generator_epoch_005.pth"),
]

# Konfigurasi Evaluasi Aktual
EVALUATION_METRICS = {
    "model_architecture": "DCGAN (Deep Convolutional Generative Adversarial Network)",
    "framework": "PyTorch",
    "image_size": f"{IMAGE_SIZE} x {IMAGE_SIZE} px",
    "latent_dimension": f"{NZ}-D (Normal Gaussian N(0, I))",
    "dataset_total_images": 1216,
    "train_images": 972,
    "test_images": 244,
    "fid_score": 2020.60,
    "pairwise_l2_diversity": 13.94,
    "mode_collapse_status": "Bebas dari Mode Collapse (Diverse Patterns)"
}

# Konfigurasi Aplikasi UI
APP_TITLE = "Batik AI Generator"
APP_SUBTITLE = "Generate New Batik Patterns with Generative AI"
APP_ICON = "🎨"
DEFAULT_NUM_IMAGES = 8
AVAILABLE_NUM_IMAGES = [4, 8, 12, 16]
