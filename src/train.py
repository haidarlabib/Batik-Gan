"""
Module: train.py
Deskripsi: Script training adversarial DCGAN untuk generasi motif batik:
           - Inisialisasi Generator dan Discriminator
           - Training loop adversarial dengan Binary Cross Entropy Loss
           - Penerapan One-Sided Label Smoothing (real=0.9, fake=0.0) untuk stabilitas
           - Evaluasi berkala dengan Fixed Latent Noise untuk memantau evolusi motif
           - Penyimpanan model checkpoint dan riwayat loss (JSON & grafik plot)
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any

# Tambahkan root path proyek
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from src.dataset import get_train_test_loaders
from src.generator import Generator
from src.discriminator import Discriminator
from src.preprocessing import denormalize

def train_dcgan(
    dataset_dir: str = "dataset",
    output_dir: str = "outputs",
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.0002,
    beta1: float = 0.5,
    nz: int = 100,
    ngf: int = 64,
    ndf: int = 64,
    sample_interval: int = 5,
    checkpoint_interval: int = 10,
    seed: int = 42,
    device_name: str = None
) -> Dict[str, Any]:
    """Menjalankan training loop DCGAN."""
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        # Optimasi utilisasi seluruh core CPU
        num_cores = os.cpu_count() or 4
        torch.set_num_threads(num_cores)
        print(f"[*] Mengatur thread CPU PyTorch ke: {num_cores} cores", flush=True)
        
    # Setup Device
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Menggunakan Device Komputasi: {device}", flush=True)
    if device.type == 'cuda':
        print(f"    GPU: {torch.cuda.get_device_name(0)}", flush=True)
        
    # Setup Direktori Output
    samples_dir = os.path.join(output_dir, "samples")
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Load Data
    print(f"[*] Menyiapkan DataLoader (Batch Size: {batch_size}, Seed: {seed})...", flush=True)
    train_loader, test_loader, train_data, test_data = get_train_test_loaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        train_ratio=0.8,
        seed=seed
    )
    print(f"[OK] Training data: {len(train_data)} citra ({len(train_loader)} batches per epoch).", flush=True)
    print(f"[OK] Test / Reference data: {len(test_data)} citra (disimpan untuk evaluasi murni).", flush=True)
    
    # Inisialisasi Model
    netG = Generator(nz=nz, ngf=ngf, nc=3).to(device)
    netD = Discriminator(nc=3, ndf=ndf).to(device)
    
    # Loss Function & Optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Fixed Noise untuk visualisasi perkembangan generasi (16 citra grid 4x4)
    fixed_noise = torch.randn(16, nz, 1, 1, device=device)
    
    # Tracking History
    history = {
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "beta1": beta1,
        "nz": nz,
        "seed": seed,
        "device": str(device),
        "train_samples_count": len(train_data),
        "test_samples_count": len(test_data),
        "loss_G_epoch": [],
        "loss_D_epoch": [],
        "D_x_epoch": [],
        "D_G_z_epoch": [],
        "epoch_times_seconds": []
    }
    
    print("\n" + "="*70)
    print("                MEMULAI TRAINING ADVERSARIAL DCGAN")
    print("="*70)
    
    start_train_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        running_loss_G = 0.0
        running_loss_D = 0.0
        running_D_x = 0.0
        running_D_G_z = 0.0
        batch_count = 0
        
        for i, real_images in enumerate(train_loader):
            b_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Label smoothing: real = 0.9, fake = 0.0
            real_labels = torch.full((b_size, 1), 0.9, dtype=torch.float, device=device)
            fake_labels = torch.full((b_size, 1), 0.0, dtype=torch.float, device=device)
            
            # -----------------------------------------------------------
            # (1) Update Discriminator: maksimalkan log(D(x)) + log(1 - D(G(z)))
            # -----------------------------------------------------------
            netD.zero_grad()
            
            # Forward real batch
            output_real = netD(real_images)
            loss_D_real = criterion(output_real, real_labels)
            loss_D_real.backward()
            d_x = output_real.mean().item()
            
            # Forward fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            output_fake = netD(fake_images.detach())  # Detach agar gradien G tidak dihitung di sini
            loss_D_fake = criterion(output_fake, fake_labels)
            loss_D_fake.backward()
            d_g_z1 = output_fake.mean().item()
            
            loss_D = loss_D_real + loss_D_fake
            optimizerD.step()
            
            # -----------------------------------------------------------
            # (2) Update Generator: maksimalkan log(D(G(z)))
            # -----------------------------------------------------------
            netG.zero_grad()
            # Generator ingin Discriminator menganggap hasil fake sebagai real (target = 1.0)
            gen_labels = torch.full((b_size, 1), 1.0, dtype=torch.float, device=device)
            output_g = netD(fake_images)
            loss_G = criterion(output_g, gen_labels)
            loss_G.backward()
            d_g_z2 = output_g.mean().item()
            optimizerG.step()
            
            # Akumulasi
            running_loss_G += loss_G.item()
            running_loss_D += loss_D.item()
            running_D_x += d_x
            running_D_G_z += d_g_z2
            batch_count += 1
            
        epoch_time = time.time() - epoch_start
        epoch_loss_G = running_loss_G / batch_count
        epoch_loss_D = running_loss_D / batch_count
        epoch_D_x = running_D_x / batch_count
        epoch_D_G_z = running_D_G_z / batch_count
        
        history["loss_G_epoch"].append(epoch_loss_G)
        history["loss_D_epoch"].append(epoch_loss_D)
        history["D_x_epoch"].append(epoch_D_x)
        history["D_G_z_epoch"].append(epoch_D_G_z)
        history["epoch_times_seconds"].append(epoch_time)
        
        print(f"Epoch [{epoch:03d}/{num_epochs:03d}] | Loss D: {epoch_loss_D:.4f} | Loss G: {epoch_loss_G:.4f} | "
              f"D(x): {epoch_D_x:.3f} | D(G(z)): {epoch_D_G_z:.3f} | Time: {epoch_time:.2f}s", flush=True)
              
        # Simpan snapshot visual berkala
        if epoch == 1 or epoch % sample_interval == 0 or epoch == num_epochs:
            with torch.no_grad():
                netG.eval()
                sample_fakes = netG(fixed_noise).detach().cpu()
                sample_fakes = denormalize(sample_fakes)
                sample_path = os.path.join(samples_dir, f"epoch_{epoch:03d}.png")
                vutils.save_image(sample_fakes, sample_path, nrow=4, padding=2, normalize=False)
                netG.train()
                print(f"    -> [Sample Saved] Snapshot disimpan ke: {sample_path}", flush=True)
                
        # Simpan checkpoint berkala
        if epoch % checkpoint_interval == 0 or epoch == num_epochs:
            ckpt_g = os.path.join(checkpoints_dir, f"generator_epoch_{epoch:03d}.pth")
            ckpt_d = os.path.join(checkpoints_dir, f"discriminator_epoch_{epoch:03d}.pth")
            torch.save(netG.state_dict(), ckpt_g)
            torch.save(netD.state_dict(), ckpt_d)
            print(f"    -> [Checkpoint Saved] Checkpoint disimpan ke: {ckpt_g}", flush=True)

    total_train_time = time.time() - start_train_time
    history["total_training_time_seconds"] = total_train_time
    print("\n" + "="*70)
    print(f"[OK] Training Selesai dalam {total_train_time:.2f} detik ({total_train_time/60:.2f} menit)!")
    print("="*70)
    
    # Simpan model final
    final_g_path = os.path.join(checkpoints_dir, "generator_final.pth")
    final_d_path = os.path.join(checkpoints_dir, "discriminator_final.pth")
    torch.save(netG.state_dict(), final_g_path)
    torch.save(netD.state_dict(), final_d_path)
    print(f"[OK] Model final Generator disimpan ke: {final_g_path}")
    
    # Simpan riwayat training JSON
    history_json_path = os.path.join(samples_dir, "training_history.json")
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        
    # Buat grafik Kurva Loss & Skor Discriminator
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Loss
    axes[0].plot(range(1, num_epochs + 1), history["loss_G_epoch"], label="Generator Loss", color='#C0392B', linewidth=2)
    axes[0].plot(range(1, num_epochs + 1), history["loss_D_epoch"], label="Discriminator Loss", color='#2980B9', linewidth=2)
    axes[0].set_title("Kurva Loss Adversarial (Generator vs Discriminator)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Plot Probabilitas Discriminator D(x) dan D(G(z))
    axes[1].plot(range(1, num_epochs + 1), history["D_x_epoch"], label="D(x) - Prob Real", color='#27AE60', linewidth=2)
    axes[1].plot(range(1, num_epochs + 1), history["D_G_z_epoch"], label="D(G(z)) - Prob Fake", color='#E67E22', linewidth=2)
    axes[1].axhline(y=0.5, color='gray', linestyle=':', label='Keseimbangan (0.5)')
    axes[1].set_title("Probabilitas Prediksi Discriminator D(x) & D(G(z))", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Probabilitas Rata-rata")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    curve_fig_path = os.path.join(samples_dir, "training_loss_curve.png")
    plt.savefig(curve_fig_path, dpi=200)
    plt.close()
    print(f"[OK] Grafik kurva training disimpan ke: {curve_fig_path}")
    
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training DCGAN Batik Generative AI")
    parser.add_argument("--epochs", type=int, default=25, help="Jumlah Epoch Training (default: 25)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch Size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning Rate Adam (default: 0.0002)")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 Adam (default: 0.5)")
    parser.add_argument("--nz", type=int, default=100, help="Dimensi Vektor Laten z (default: 100)")
    parser.add_argument("--sample-interval", type=int, default=5, help="Interval simpan sampel citra (default: 5)")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Interval simpan checkpoint (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed reproduktibilitas (default: 42)")
    
    args = parser.parse_args()
    
    train_dcgan(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta1=args.beta1,
        nz=args.nz,
        sample_interval=args.sample_interval,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed
    )
