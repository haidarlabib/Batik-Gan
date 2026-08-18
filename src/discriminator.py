"""
Module: discriminator.py
Deskripsi: Arsitektur Discriminator DCGAN (Deep Convolutional GAN) berbasis PyTorch:
           - Mengklasifikasikan citra input (64x64x3) sebagai citra Real (Asli) atau Fake (Sintetis)
           - Menggunakan Strided Convolution (Downsampling), Batch Normalization, dan LeakyReLU (0.2)
           - Lapisan output menghasilkan probabilitas skalar [0, 1] melalui aktivasi Sigmoid
           - Inisialisasi bobot standar DCGAN (Normal(0.0, 0.02))
"""

import os
import sys

# Tambahkan root path proyek agar import src.* selalu berhasil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
try:
    from src.generator import weights_init
except (ImportError, ModuleNotFoundError):
    from generator import weights_init

class Discriminator(nn.Module):
    """Discriminator Network untuk DCGAN 64x64."""
    
    def __init__(self, nc: int = 3, ndf: int = 64):
        """
        Args:
            nc: Jumlah channel input citra (default 3 untuk RGB)
            ndf: Ukuran filter dasar Discriminator (default 64)
        """
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        
        self.main = nn.Sequential(
            # Input: (nc) x 64 x 64 -> Output: (ndf) x 32 x 32 (64 x 32 x 32)
            # Tanpa BatchNorm pada lapisan pertama Discriminator (sesuai paper DCGAN)
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf) x 32 x 32 -> (ndf*2) x 16 x 16 (128 x 16 x 16)
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*2) x 16 x 16 -> (ndf*4) x 8 x 8 (256 x 8 x 8)
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*4) x 8 x 8 -> (ndf*8) x 4 x 4 (512 x 4 x 4)
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*8) x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        
        # Terapkan inisialisasi bobot
        self.apply(weights_init)
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            input_tensor: Tensor citra (B, nc, 64, 64)
        Returns:
            Tensor skalar probabilitas Real (B, 1, 1, 1) -> di-flatten menjadi (B, 1)
        """
        out = self.main(input_tensor)
        return out.view(-1, 1)

if __name__ == "__main__":
    netD = Discriminator(nc=3, ndf=64)
    dummy_images = torch.randn(8, 3, 64, 64)
    probs = netD(dummy_images)
    print(f"Discriminator Architecture:\n{netD}")
    print(f"Input Shape : {dummy_images.shape}")
    print(f"Output Shape: {probs.shape}")
    print(f"Output Range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
