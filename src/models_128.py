"""
Module: models_128.py
Deskripsi: Arsitektur Model Generatif 128x128 untuk Peningkatan Kualitas Citra Motif Batik:
            1. Improved DCGAN 128x128 (6-layer Transposed Conv dengan Spectral Norm & BatchNorm)
            2. StyleGAN2-ADA 128x128 (Mapping Network, Style Modulated Convolutions, ADA Augmentation & R1 Penalty)
"""

import math
import random
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    """Inisialisasi bobot standar normal (0.0, 0.02) untuk stabilitas GAN."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# =============================================================================
# 1. IMPROVED DCGAN 128x128
# =============================================================================

class ImprovedDCGANGenerator128(nn.Module):
    """Generator DCGAN 128x128 dengan 6 tahap upsampling."""
    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3):
        super(ImprovedDCGANGenerator128, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        
        self.main = nn.Sequential(
            # z: (nz x 1 x 1) -> (ngf*16 x 4 x 4) [1024 x 4 x 4]
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            
            # (ngf*16 x 4 x 4) -> (ngf*8 x 8 x 8) [512 x 8 x 8]
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # (ngf*8 x 8 x 8) -> (ngf*4 x 16 x 16) [256 x 16 x 16]
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # (ngf*4 x 16 x 16) -> (ngf*2 x 32 x 32) [128 x 32 x 32]
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # (ngf*2 x 32 x 32) -> (ngf x 64 x 64) [64 x 64 x 64]
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # (ngf x 64 x 64) -> (nc x 128 x 128) [3 x 128 x 128]
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, self.nz, 1, 1)
        return self.main(x)


class ImprovedDCGANDiscriminator128(nn.Module):
    """Discriminator DCGAN 128x128 dengan Spectral Normalization."""
    def __init__(self, nc: int = 3, ndf: int = 64):
        super(ImprovedDCGANDiscriminator128, self).__init__()
        self.main = nn.Sequential(
            # (nc x 128 x 128) -> (ndf x 64 x 64)
            nn.utils.spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (ndf x 64 x 64) -> (ndf*2 x 32 x 32)
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (ndf*2 x 32 x 32) -> (ndf*4 x 16 x 16)
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (ndf*4 x 16 x 16) -> (ndf*8 x 8 x 8)
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (ndf*8 x 8 x 8) -> (ndf*16 x 4 x 4)
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (ndf*16 x 4 x 4) -> (1 x 1 x 1)
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(-1, 1)


# =============================================================================
# 2. STYLEGAN2-ADA (Adaptive Discriminator Augmentation) 128x128
# =============================================================================

class PixelNorm(nn.Module):
    """Pixelwise Feature Vector Normalization."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class MappingNetwork(nn.Module):
    """Mapping Network MLP: z in R^100 -> w in R^w_dim."""
    def __init__(self, z_dim: int = 100, w_dim: int = 256, num_layers: int = 3):
        super(MappingNetwork, self).__init__()
        layers = [PixelNorm()]
        for i in range(num_layers):
            in_f = z_dim if i == 0 else w_dim
            layers.append(nn.Linear(in_f, w_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 4:
            z = z.view(z.size(0), -1)
        return self.net(z)


class StyleBlock128(nn.Module):
    """Synthesis Style Block dengan Style-Modulation & Upsampling."""
    def __init__(self, in_c: int, out_c: int, w_dim: int = 256, upsample: bool = True):
        super(StyleBlock128, self).__init__()
        self.upsample = upsample
        self.style_affine1 = nn.Linear(w_dim, in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        
        self.style_affine2 = nn.Linear(w_dim, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            
        # Style modulation 1
        s1 = self.style_affine1(w).unsqueeze(-1).unsqueeze(-1)
        x = self.conv1(x * (s1 + 1.0))
        x = self.bn1(x)
        x = self.act(x)
        
        # Style modulation 2
        s2 = self.style_affine2(w).unsqueeze(-1).unsqueeze(-1)
        x = self.conv2(x * (s2 + 1.0))
        x = self.bn2(x)
        x = self.act(x)
        return x


class StyleGAN2ADAGenerator128(nn.Module):
    """
    StyleGAN2-ADA Generator 128x128:
    - Mapping Network (z -> w)
    - Multi-scale Style Synthesis Blocks (4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128)
    - Tanh output di rentang [-1, 1]
    """
    def __init__(self, z_dim: int = 100, w_dim: int = 256, nc: int = 3):
        super(StyleGAN2ADAGenerator128, self).__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.nc = nc
        
        self.mapping = MappingNetwork(z_dim=z_dim, w_dim=w_dim, num_layers=3)
        
        # Learned constant 256 x 4 x 4
        self.const = nn.Parameter(torch.randn(1, 256, 4, 4))
        
        # Style Blocks
        self.block4 = StyleBlock128(256, 256, w_dim=w_dim, upsample=False)
        self.block8 = StyleBlock128(256, 128, w_dim=w_dim, upsample=True)    # 8x8
        self.block16 = StyleBlock128(128, 64, w_dim=w_dim, upsample=True)   # 16x16
        self.block32 = StyleBlock128(64, 32, w_dim=w_dim, upsample=True)    # 32x32
        self.block64 = StyleBlock128(32, 16, w_dim=w_dim, upsample=True)    # 64x64
        self.block128 = StyleBlock128(16, 16, w_dim=w_dim, upsample=True)   # 128x128
        
        # ToRGB
        self.to_rgb = nn.Sequential(
            nn.Conv2d(16, nc, kernel_size=1),
            nn.Tanh()
        )
        self.apply(weights_init)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        w = self.mapping(z)
        x = self.const.repeat(b, 1, 1, 1)
        
        x = self.block4(x, w)
        x = self.block8(x, w)
        x = self.block16(x, w)
        x = self.block32(x, w)
        x = self.block64(x, w)
        x = self.block128(x, w)
        
        rgb = self.to_rgb(x)
        return rgb


class AdaptiveAugmenter(nn.Module):
    """
    Adaptive Discriminator Augmentation (ADA) Module:
    Menerapkan transformasi diferensiabel dengan probabilitas p pada citra sebelum masuk Discriminator.
    """
    def __init__(self, p: float = 0.0):
        super(AdaptiveAugmenter, self).__init__()
        self.p = p
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0 or not self.training:
            return x
            
        b, c, h, w = x.shape
        
        # 1. Random Horizontal Flip with prob p
        if random.random() < self.p:
            x = torch.flip(x, dims=[3])
            
        # 2. Random 90/180/270 Rotation with prob p
        if random.random() < self.p:
            k = random.choice([1, 2, 3])
            x = torch.rot90(x, k, dims=[2, 3])
            
        # 3. Random Translation with prob p
        if random.random() < self.p:
            dx = random.randint(-4, 4)
            dy = random.randint(-4, 4)
            x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
            
        return x
