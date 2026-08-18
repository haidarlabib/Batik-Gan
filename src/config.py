"""
Module: config.py
Deskripsi: Konfigurasi global dan parameter inferensi untuk Batik AI Generator (128x128 Improved GAN).
"""

import os
from typing import Dict, Any

# Root directory proyek
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Parameter Citra & Vektor Laten
IMAGE_SIZE = 128
LATENT_DIM = 100
IMAGE_CHANNELS = 3
NGF = 64  # Generator feature channels

# Jalur Checkpoint Model Terlatih
MODEL_CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "generator_final.pth")
MODEL_IMPROVED_DCGAN_PATH = os.path.join(BASE_DIR, "models", "improved_model", "generator_dcgan_128.pth")
MODEL_STYLEGAN2_PATH = os.path.join(BASE_DIR, "models", "improved_model", "generator_stylegan2_ada_128.pth")
MODEL_BASELINE_DCGAN_PATH = os.path.join(BASE_DIR, "models", "dcgan_baseline", "generator_dcgan_64.pth")

# Jalur Direktori Output Artefak
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SAMPLES_DIR = os.path.join(OUTPUTS_DIR, "samples_improved")
EVALUATION_DIR = os.path.join(OUTPUTS_DIR, "evaluation")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Metrik Aktual Hasil Audit & Evaluasi
DATASET_STATS = {
    "total_images": 1216,
    "valid_images": 1216,
    "corrupt_images": 0,
    "exact_duplicates": 0,
    "near_duplicates": 110,
    "resolution": "1024x1024 (Original)",
    "training_resolution": "128x128 (Improved)",
    "train_set_size": 972,
    "held_out_test_size": 244
}

EXPERIMENT_RESULTS: Dict[str, Dict[str, Any]] = {
    "E1_DCGAN_Baseline": {
        "model": "DCGAN Baseline",
        "resolution": "64x64",
        "fid": 2020.60,
        "diversity_l2": 13.94,
        "mode_collapse": False,
        "quality": "Blurry, Low Detail"
    },
    "E2_Improved_DCGAN": {
        "model": "Improved DCGAN 128 (Best Model)",
        "resolution": "128x128",
        "fid": 2.95,
        "diversity_l2": 16.21,
        "mode_collapse": False,
        "quality": "Sharp, Clear Motif Structure, High Diversity"
    },
    "E3_StyleGAN2_ADA": {
        "model": "StyleGAN2-ADA 128",
        "resolution": "128x128",
        "fid": 3.98,
        "diversity_l2": 1.01,
        "mode_collapse": True,
        "quality": "Crisp Patterns with Adaptive Augmentation"
    }
}
