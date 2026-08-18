"""
Module: preprocessing.py
Deskripsi: Pipeline transformasi dan normalisasi citra batik untuk arsitektur DCGAN:
           - Validasi format RGB
           - Resize ke resolusi target (64x64)
           - Konversi ke PyTorch Tensor
           - Normalisasi ke rentang [-1, 1] sesuai aktivasi Tanh pada Generator
           - Fungsi denormalisasi untuk rekonstruksi citra visual [0, 1]
"""

import torch
from torchvision import transforms
from PIL import Image

def get_transforms(image_size: int = 64) -> transforms.Compose:
    """Mengembalikan pipeline transformasi standar untuk DCGAN."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Mengembalikan tensor dari rentang [-1, 1] ke rentang visual [0, 1].
    
    Args:
        tensor: PyTorch Tensor berbentuk (C, H, W) atau (B, C, H, W)
    Returns:
        Tensor ter-klip di rentang [0, 1]
    """
    # x_denorm = (x * std) + mean = (x * 0.5) + 0.5
    denorm = tensor * 0.5 + 0.5
    return torch.clamp(denorm, 0.0, 1.0)

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Mengonversi single tensor (3, H, W) di rentang [-1, 1] menjadi PIL Image."""
    denorm = denormalize(tensor.cpu().detach())
    if denorm.dim() == 4:
        denorm = denorm[0]
    np_img = denorm.permute(1, 2, 0).numpy()
    np_img = (np_img * 255.0).astype('uint8')
    return Image.fromarray(np_img)
