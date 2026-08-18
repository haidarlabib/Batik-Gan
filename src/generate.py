"""
Module: generate.py
Deskripsi: Modul inferensi untuk menghasilkan motif batik sintetis baru:
           - Memuat bobot Generator terlatih
           - Menghasilkan sejumlah N citra motif batik baru dari vektor laten acak
           - Menyimpan citra individual (batik_01.png, batik_02.png, ...) dan grid montage
           - Dapat dipanggil via CLI maupun diimpor sebagai fungsi oleh Notebook / Web App
"""

import os
import sys
import argparse
from typing import List, Optional
from PIL import Image

# Tambahkan root path proyek
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.utils as vutils
try:
    from src.generator import Generator
    from src.preprocessing import denormalize, tensor_to_pil
except (ImportError, ModuleNotFoundError):
    from generator import Generator
    from preprocessing import denormalize, tensor_to_pil

def generate_images(
    num_images: int = 16,
    model_path: str = "outputs/checkpoints/generator_final.pth",
    output_dir: str = "outputs/generated",
    nz: int = 100,
    ngf: int = 64,
    seed: Optional[int] = None,
    device_name: Optional[str] = None
) -> List[str]:
    """
    Menghasilkan N citra motif batik sintetis dan menyimpannya ke disk.
    
    Args:
        num_images: Jumlah citra yang ingin dibuat
        model_path: Path ke file checkpoint Generator (.pth)
        output_dir: Folder tujuan penyimpanan citra
        nz: Dimensi vektor laten
        ngf: Filter size Generator
        seed: Random seed opsional untuk reproduktibilitas
        device_name: 'cuda', 'cpu', atau None (otomatis)
        
    Returns:
        List path file citra yang berhasil dihasilkan
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Cari checkpoint generator
    if not os.path.exists(model_path):
        ckpts = sorted([f for f in os.listdir("outputs/checkpoints") if f.startswith("generator_epoch")])
        if ckpts:
            model_path = os.path.join("outputs/checkpoints", ckpts[-1])
        else:
            raise FileNotFoundError(f"File model '{model_path}' tidak ditemukan.")
            
    netG = Generator(nz=nz, ngf=ngf, nc=3).to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    
    print(f"[*] Menghasilkan {num_images} motif batik sintetis menggunakan model: '{model_path}'...")
    noise = torch.randn(num_images, nz, 1, 1, device=device)
    
    with torch.no_grad():
        fake_tensors = netG(noise).detach().cpu()
        fake_denorm = denormalize(fake_tensors)
        
    saved_paths = []
    
    # 1. Simpan citra individual
    for i in range(num_images):
        pil_img = tensor_to_pil(fake_tensors[i])
        fname = f"batik_{i+1:03d}.png"
        fpath = os.path.join(output_dir, fname)
        pil_img.save(fpath)
        saved_paths.append(fpath)
        
    # 2. Simpan grid montage
    grid_path = os.path.join(output_dir, "batik_grid_montage.png")
    nrow = 4 if num_images >= 16 else (2 if num_images >= 4 else 1)
    vutils.save_image(fake_denorm, grid_path, nrow=nrow, padding=2, normalize=False)
    
    print(f"[OK] {num_images} citra batik berhasil dibuat dan disimpan di folder '{output_dir}'.")
    print(f"[OK] Grid montage disimpan di '{grid_path}'.")
    return saved_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Citra Motif Batik Sintetis")
    parser.add_argument("--num", type=int, default=16, help="Jumlah citra batik yang ingin dihasilkan (default: 16)")
    parser.add_argument("--model", type=str, default="outputs/checkpoints/generator_final.pth", help="Path model Generator .pth")
    parser.add_argument("--outdir", type=str, default="outputs/generated", help="Folder output")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (opsional)")
    args = parser.parse_args()
    
    generate_images(num_images=args.num, model_path=args.model, output_dir=args.outdir, seed=args.seed)
