"""
Module: config.py
Deskripsi: Konfigurasi global, parameter model, jalur checkpoint, dan metadata evaluasi aktual untuk Batik AI Generator.
"""

import os
import json
from typing import Dict, Any

# Root directory proyek (relatif terhadap direktori file config.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Parameter Citra & Vektor Laten
IMAGE_SIZE = 128
LATENT_DIM = 100
IMAGE_CHANNELS = 3
NGF = 64  # Generator feature channels

# Jalur Checkpoint Model Terlatih (Relative Path)
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_CHECKPOINT_PATH = os.path.join(MODELS_DIR, "generator_final.pth")
MODEL_IMPROVED_DCGAN_PATH = os.path.join(MODELS_DIR, "improved_model", "generator_dcgan_128.pth")
MODEL_STYLEGAN2_PATH = os.path.join(MODELS_DIR, "improved_model", "generator_stylegan2_ada_128.pth")
MODEL_BASELINE_DCGAN_PATH = os.path.join(MODELS_DIR, "dcgan_baseline", "generator_dcgan_64.pth")

# Jalur Direktori Output Artefak & Dataset
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SAMPLES_DIR = os.path.join(OUTPUTS_DIR, "samples_improved")
EVALUATION_DIR = os.path.join(OUTPUTS_DIR, "evaluation")
AUDIT_DIR = os.path.join(OUTPUTS_DIR, "audit")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# -----------------------------------------------------------------------------
# Metadata Dataset Aktual (Single Source of Truth)
# -----------------------------------------------------------------------------
DATASET_STATS: Dict[str, Any] = {
    "total_images": 1216,
    "valid_images": 1216,
    "corrupt_images": 0,
    "exact_duplicates": 0,
    "near_duplicates": 110,
    "resolution": "1024x1024 (Original)",
    "training_resolution": "128x128 (Improved Native)",
    "train_set_size": 972,
    "held_out_test_size": 244,
    "num_base_ids": 608,
    "brightness_mean": 141.03,
    "sharpness_laplacian_mean": 3848.28,
    "augmentation": "Dihedral D4 Symmetries (H-Flip, V-Flip, 90/180/270 Rotations)"
}

# -----------------------------------------------------------------------------
# Hasil Evaluasi Aktual Eksperimen (Single Source of Truth)
# -----------------------------------------------------------------------------
EXPERIMENT_RESULTS: Dict[str, Dict[str, Any]] = {
    "E1_DCGAN_Baseline": {
        "model": "DCGAN Baseline",
        "architecture": "DCGAN 5-layer Transposed Conv",
        "resolution": "64x64",
        "fid": 2020.60,
        "diversity_l2": 13.94,
        "mode_collapse": False,
        "quality": "Blurry, Low Detail (Keterbatasan resolusi 64x64)"
    },
    "E2_Improved_DCGAN": {
        "model": "Improved DCGAN 128 (Best Model)",
        "architecture": "Improved DCGAN 6-layer Transposed Conv + Spectral Norm",
        "resolution": "128x128",
        "fid": 2.95,
        "diversity_l2": 16.21,
        "mode_collapse": False,
        "quality": "Sharp, Clear Motif Structure, High Diversity (Juara FID & Keragaman)"
    },
    "E3_StyleGAN2_ADA": {
        "model": "StyleGAN2-ADA 128",
        "architecture": "StyleGAN2-ADA (Mapping Network + Style Modulation + ADA)",
        "resolution": "128x128",
        "fid": 3.98,
        "diversity_l2": 1.01,
        "mode_collapse": True,
        "quality": "Crisp Patterns with Adaptive Augmentation"
    }
}

# Eksport simbol publik
__all__ = [
    "BASE_DIR",
    "IMAGE_SIZE",
    "LATENT_DIM",
    "IMAGE_CHANNELS",
    "NGF",
    "MODELS_DIR",
    "MODEL_CHECKPOINT_PATH",
    "MODEL_IMPROVED_DCGAN_PATH",
    "MODEL_STYLEGAN2_PATH",
    "MODEL_BASELINE_DCGAN_PATH",
    "OUTPUTS_DIR",
    "SAMPLES_DIR",
    "EVALUATION_DIR",
    "AUDIT_DIR",
    "DATASET_DIR",
    "DATASET_STATS",
    "EXPERIMENT_RESULTS",
]
