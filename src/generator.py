"""
Module: generator.py
Deskripsi: Arsitektur Generator DCGAN (Deep Convolutional GAN) berbasis PyTorch:
           - Memetakan vektor laten acak z (100-D) menjadi citra sintetis motif batik RGB (64x64x3)
           - Menggunakan lapisan Transposed Convolution (Upsampling), Batch Normalization, dan ReLU
           - Lapisan output menggunakan aktivasi Tanh untuk menghasilkan piksel di rentang [-1, 1]
           - Inisialisasi bobot standar DCGAN (Normal(0.0, 0.02))
"""

import torch
import torch.nn as nn

def weights_init(m):
    """Inisialisasi bobot kustom sesuai standar paper DCGAN (Radford et al.)."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    """Generator Network untuk DCGAN 64x64."""
    
    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3):
        """
        Args:
            nz: Dimensi vektor laten z (default 100)
            ngf: Ukuran filter dasar Generator (default 64)
            nc: Jumlah channel output (default 3 untuk RGB)
        """
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        
        self.main = nn.Sequential(
            # Input: Latent vector z (nz x 1 x 1) -> Output: (ngf*8) x 4 x 4 (512 x 4 x 4)
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # State size: (ngf*8) x 4 x 4 -> (ngf*4) x 8 x 8 (256 x 8 x 8)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # State size: (ngf*4) x 8 x 8 -> (ngf*2) x 16 x 16 (128 x 16 x 16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # State size: (ngf*2) x 16 x 16 -> (ngf) x 32 x 32 (64 x 32 x 32)
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # State size: (ngf) x 32 x 32 -> (nc) x 64 x 64 (3 x 64 x 64)
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        
        # Terapkan inisialisasi bobot
        self.apply(weights_init)
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            input_tensor: Tensor berbentuk (B, nz, 1, 1) atau (B, nz)
        Returns:
            Tensor citra sintetis (B, nc, 64, 64) di rentang [-1, 1]
        """
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.view(-1, self.nz, 1, 1)
        return self.main(input_tensor)

if __name__ == "__main__":
    netG = Generator(nz=100, ngf=64, nc=3)
    dummy_noise = torch.randn(8, 100, 1, 1)
    fake_images = netG(dummy_noise)
    print(f"Generator Architecture:\n{netG}")
    print(f"Input Shape : {dummy_noise.shape}")
    print(f"Output Shape: {fake_images.shape}")
    print(f"Output Range: [{fake_images.min().item():.2f}, {fake_images.max().item():.2f}]")
