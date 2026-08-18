"""
Module: inference.py
Deskripsi: Engine inferensi multi-arsitektur untuk generasi motif batik sintetis (128x128 & 64x64).
"""

import os
import io
import zipfile
from typing import List, Optional, Tuple

import torch
import numpy as np
from PIL import Image

from src.models_128 import ImprovedDCGANGenerator128, StyleGAN2ADAGenerator128
from src.generator import Generator as BaselineDCGANGenerator64
from src.config import (
    MODEL_CHECKPOINT_PATH,
    MODEL_IMPROVED_DCGAN_PATH,
    MODEL_STYLEGAN2_PATH,
    MODEL_BASELINE_DCGAN_PATH,
    IMAGE_SIZE,
    LATENT_DIM
)

def get_device() -> torch.device:
    """Mendeteksi ketersediaan hardware GPU (CUDA) atau CPU Multi-Core."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_generator(
    model_type: str = "improved_dcgan",
    checkpoint_path: Optional[str] = None
) -> Tuple[torch.nn.Module, int]:
    """
    Memuat arsitektur Generator berdasarkan pilihan model:
    - 'improved_dcgan': Improved DCGAN 128x128 (Default Champion)
    - 'stylegan2_ada': StyleGAN2-ADA 128x128
    - 'dcgan_baseline': DCGAN Baseline 64x64
    """
    device = get_device()
    
    if model_type == "stylegan2_ada":
        gen = StyleGAN2ADAGenerator128(z_dim=100, w_dim=128, nc=3)
        res = 128
        ckpt = checkpoint_path or MODEL_STYLEGAN2_PATH
        if not os.path.exists(ckpt):
            ckpt = MODEL_CHECKPOINT_PATH
    elif model_type == "dcgan_baseline":
        gen = BaselineDCGANGenerator64(nz=100, ngf=64, nc=3)
        res = 64
        ckpt = checkpoint_path or MODEL_BASELINE_DCGAN_PATH
        if not os.path.exists(ckpt):
            ckpt = os.path.join(os.path.dirname(MODEL_CHECKPOINT_PATH), "generator_final.pth")
    else:  # 'improved_dcgan' default
        gen = ImprovedDCGANGenerator128(nz=100, ngf=64, nc=3)
        res = 128
        ckpt = checkpoint_path or MODEL_IMPROVED_DCGAN_PATH
        if not os.path.exists(ckpt):
            ckpt = MODEL_CHECKPOINT_PATH
            
    if os.path.exists(ckpt):
        try:
            state = torch.load(ckpt, map_location=device, weights_only=True)
            gen.load_state_dict(state)
        except Exception:
            state = torch.load(ckpt, map_location=device)
            gen.load_state_dict(state)
            
    gen.to(device)
    gen.eval()
    return gen, res

def generate_batik_images(
    generator: torch.nn.Module,
    num_images: int = 8,
    seed: Optional[int] = None,
    latent_dim: int = LATENT_DIM
) -> List[Image.Image]:
    """
    Menghasilkan N citra motif batik sintetis berformat PIL Image.
    Menggunakan Tanh output [-1, 1] yang didenormalisasi ke [0, 255].
    """
    device = next(generator.parameters()).device
    
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, device=device)
        fake_tensors = generator(z)
        
        # Denormalisasi [-1, 1] ke [0, 1]
        fake_tensors = (fake_tensors + 1.0) / 2.0
        fake_tensors = torch.clamp(fake_tensors, 0.0, 1.0)
        
        pil_images = []
        for i in range(num_images):
            img_np = fake_tensors[i].cpu().permute(1, 2, 0).numpy()
            img_uint8 = (img_np * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)
            pil_images.append(pil_img)
            
    return pil_images

def create_zip_package(
    images: List[Image.Image],
    seed: Optional[int] = None,
    resolution: int = 128
) -> io.BytesIO:
    """Mengemas seluruh citra yang dihasilkan ke dalam memory buffer ZIP."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for idx, img in enumerate(images):
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_bytes = img_byte_arr.getvalue()
            seed_tag = f"_seed_{seed}" if seed is not None else ""
            zip_file.writestr(f"batik_{resolution}x{resolution}_{idx+1:03d}{seed_tag}.png", img_bytes)
            
    zip_buffer.seek(0)
    return zip_buffer
