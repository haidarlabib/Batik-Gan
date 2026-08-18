"""
Module: inference.py
Deskripsi: Engine inferensi dan utilitas generasi citra batik sintetis untuk Streamlit Deployment:
            - Memuat bobot Generator terlatih dari checkpoint
            - Menghasilkan sejumlah N citra batik dari vektor laten acak atau berdasar seed
            - Mengonversi tensor menjadi PIL Image dan PNG buffer biner
            - Mengemas citra hasil generasi ke dalam file ZIP untuk kemudahan download kolektif
"""

import os
import io
import zipfile
from typing import List, Tuple, Optional
from PIL import Image

import torch
from src.generator import Generator
from src.preprocessing import denormalize, tensor_to_pil
from src.config import NZ, NGF, NC, CANDIDATE_MODEL_PATHS

def get_device() -> torch.device:
    """Mengembalikan device komputasi yang tersedia (CUDA GPU jika ada, fallback CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_checkpoint_path() -> Optional[str]:
    """Mencari file checkpoint Generator yang valid dari daftar kandidat."""
    for path in CANDIDATE_MODEL_PATHS:
        if os.path.exists(path):
            return path
    return None

def load_generator(checkpoint_path: Optional[str] = None, device: Optional[torch.device] = None) -> Generator:
    """
    Memuat model Generator DCGAN dari checkpoint.
    
    Args:
        checkpoint_path: Path ke file .pth (jika None, otomatis mencari)
        device: Device komputasi PyTorch (jika None, otomatis deteksi)
    Returns:
        Instance Generator dalam mode evaluasi (eval)
    Raises:
        FileNotFoundError: Jika tidak ada checkpoint yang ditemukan
    """
    if device is None:
        device = get_device()
        
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        checkpoint_path = find_checkpoint_path()
        if checkpoint_path is None:
            raise FileNotFoundError(
                "Generator model not found. Please make sure the trained model checkpoint exists in the expected models directory."
            )
            
    generator = Generator(nz=NZ, ngf=NGF, nc=NC).to(device)
    
    # Load bobot
    state_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator

def generate_batik_images(
    generator: Generator,
    num_images: int = 8,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None
) -> List[Tuple[Image.Image, bytes]]:
    """
    Menghasilkan sejumlah citra batik sintetis baru.
    
    Args:
        generator: Model Generator DCGAN yang telah diload
        num_images: Jumlah citra yang dihasilkan (misal 4, 8, 12, 16)
        seed: Random seed integer opsional untuk reproduktibilitas
        device: Device komputasi
    Returns:
        List of Tuple (PIL Image, PNG Bytes)
    """
    if device is None:
        device = get_device()
        
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    # Buat vektor laten acak z ~ N(0, I)
    z = torch.randn(num_images, NZ, 1, 1, device=device)
    
    with torch.no_grad():
        fake_tensors = generator(z).detach().cpu()
        
    results = []
    for i in range(num_images):
        pil_img = tensor_to_pil(fake_tensors[i])
        
        # Simpan ke byte buffer PNG
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        
        results.append((pil_img, png_bytes))
        
    return results

def create_zip_package(
    image_items: List[Tuple[Image.Image, bytes]],
    prefix: str = "batik_synthetic"
) -> bytes:
    """
    Mengemas seluruh citra hasil generasi ke dalam satu file ZIP in-memory.
    
    Args:
        image_items: List of Tuple (PIL.Image, png_bytes)
        prefix: Prefix penamaan file dalam ZIP
    Returns:
        Bytes buffer dari file ZIP
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for idx, (_, png_bytes) in enumerate(image_items):
            fname = f"{prefix}_{idx+1:02d}.png"
            zip_file.writestr(fname, png_bytes)
    return zip_buffer.getvalue()
