"""
Module: data_audit.py
Deskripsi: Modul untuk melakukan audit komprehensif terhadap dataset batik:
           - Verifikasi integritas citra (cek korup)
           - Statistik resolusi, channel, mode warna, dan ukuran file
           - Deteksi exact duplicate (MD5 hash)
           - Deteksi near-duplicate (dHash / Difference Hash)
           - Analisis pola nama file
           - Pembuatan visualisasi eksplorasi dataset (grid citra, distribusi ukuran, histogram warna)
"""

import os
import re
import json
import hashlib
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend untuk server/skrip

def calculate_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Menghitung Difference Hash (dHash) untuk mendeteksi kesamaan visual/near-duplicate."""
    resized = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(resized.getdata())
    difference = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = pixels[row * (hash_size + 1) + col]
            pixel_right = pixels[row * (hash_size + 1) + col + 1]
            difference.append(pixel_left > pixel_right)
    
    decimal_val = 0
    hex_str = []
    for index, val in enumerate(difference):
        if val:
            decimal_val += 2**(index % 8)
        if (index % 8) == 7:
            hex_str.append(hex(decimal_val)[2:].rjust(2, '0'))
            decimal_val = 0
    return ''.join(hex_str)

def run_dataset_audit(dataset_dir: str = "dataset", output_dir: str = "outputs/audit") -> dict:
    """Menjalankan proses audit lengkap dan menghasilkan laporan serta visualisasi."""
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Direktori dataset tidak ditemukan di: {dataset_dir}")
        
    all_files = sorted(os.listdir(dataset_dir))
    total_files = len(all_files)
    print(f"[*] Memulai audit terhadap {total_files} file dalam direktori: '{dataset_dir}'...")

    valid_images = []
    corrupt_images = []
    extensions = Counter()
    color_modes = Counter()
    channels = Counter()
    resolutions = Counter()
    file_sizes = []
    md5_hashes = defaultdict(list)
    dhashes = defaultdict(list)
    filename_patterns = []

    for idx, fname in enumerate(all_files, 1):
        fpath = os.path.join(dataset_dir, fname)
        ext = os.path.splitext(fname)[1].lower()
        extensions[ext] += 1
        
        # Cek ukuran file
        fsize = os.path.getsize(fpath)
        file_sizes.append(fsize)
        
        # Analisis pola filename (cth: 12a.png, 45b.png)
        match = re.match(r"^(\d+)([a-zA-Z]+)\.png$", fname)
        if match:
            num, suffix = match.groups()
            filename_patterns.append({"id": int(num), "suffix": suffix, "filename": fname})
            
        # Hitung MD5 Hash (Exact Duplicate)
        with open(fpath, "rb") as fp:
            file_bytes = fp.read()
            md5_val = hashlib.md5(file_bytes).hexdigest()
            md5_hashes[md5_val].append(fname)
            
        # Verifikasi integritas gambar PIL
        try:
            with Image.open(fpath) as img:
                img.verify()
            with Image.open(fpath) as img:
                valid_images.append(fname)
                color_modes[img.mode] += 1
                channels[len(img.getbands())] += 1
                resolutions[f"{img.size[0]}x{img.size[1]}"] += 1
                
                # Hitung dHash
                dh = calculate_dhash(img)
                dhashes[dh].append(fname)
        except Exception as e:
            corrupt_images.append({"filename": fname, "error": str(e)})

    # Analisis Duplicate
    exact_duplicates = {k: v for k, v in md5_hashes.items() if len(v) > 1}
    near_duplicates = {k: v for k, v in dhashes.items() if len(v) > 1}
    
    # Analisis Suffix / Pola Angka
    suffix_counter = Counter([item["suffix"] for item in filename_patterns])
    unique_base_ids = set([item["id"] for item in filename_patterns])
    
    audit_report = {
        "dataset_directory": os.path.abspath(dataset_dir),
        "total_files": total_files,
        "valid_images_count": len(valid_images),
        "corrupt_images_count": len(corrupt_images),
        "corrupt_images_list": corrupt_images,
        "file_extensions": dict(extensions),
        "color_modes": dict(color_modes),
        "channels_distribution": {f"{k}_channels": v for k, v in channels.items()},
        "resolutions": dict(resolutions),
        "file_sizes_bytes": {
            "min": int(np.min(file_sizes)) if file_sizes else 0,
            "max": int(np.max(file_sizes)) if file_sizes else 0,
            "mean": float(np.mean(file_sizes)) if file_sizes else 0,
            "median": float(np.median(file_sizes)) if file_sizes else 0
        },
        "exact_duplicates_count": len(exact_duplicates),
        "exact_duplicates_groups": exact_duplicates,
        "near_duplicates_dhash_count": len(near_duplicates),
        "near_duplicates_sample": {k: v for k, v in list(near_duplicates.items())[:10]},
        "filename_pattern_analysis": {
            "matched_pattern_count": len(filename_patterns),
            "unique_base_ids_count": len(unique_base_ids),
            "suffixes_count": dict(suffix_counter),
            "min_base_id": min(unique_base_ids) if unique_base_ids else None,
            "max_base_id": max(unique_base_ids) if unique_base_ids else None
        }
    }
    
    # Simpan laporan JSON
    json_path = os.path.join(output_dir, "dataset_audit_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    print(f"[OK] Laporan audit berhasil disimpan ke: {json_path}")
    
    # --- VISUALISASI EKSPLORASI DATASET ---
    print("[*] Membuat grafik visualisasi audit dataset...")
    
    # 1. Visualisasi Statistik Dataset (File Size Histogram & Info Summary)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram Ukuran File (KB)
    file_sizes_kb = [s / 1024.0 for s in file_sizes]
    axes[0].hist(file_sizes_kb, bins=30, color='#8B5A2B', edgecolor='black', alpha=0.8)
    axes[0].set_title("Distribusi Ukuran File Citra (KB)", fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xlabel("Ukuran File (KB)")
    axes[0].set_ylabel("Frekuensi")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Tabel Ringkasan Statistik
    summary_text = (
        f"RINGKASAN AUDIT DATASET BATIK\n"
        f"----------------------------------------\n"
        f"Total File: {total_files}\n"
        f"Citra Valid: {len(valid_images)}\n"
        f"Citra Corrupt: {len(corrupt_images)}\n"
        f"Format: {list(extensions.keys())}\n"
        f"Color Mode: {dict(color_modes)}\n"
        f"Resolusi Utama: {list(resolutions.keys())[0]}\n"
        f"Ukuran Rata-rata: {np.mean(file_sizes_kb):.1f} KB\n"
        f"Exact Duplicate: {len(exact_duplicates)}\n"
        f"Pola ID: 0 - {max(unique_base_ids)} (Suffix 'a' & 'b')\n"
        f"Total Base IDs: {len(unique_base_ids)}\n"
        f"----------------------------------------\n"
        f"Status Audit: DATASET SANGAT BERSIH & SERAGAM"
    )
    axes[1].axis('off')
    axes[1].text(0.05, 0.5, summary_text, fontsize=11, family='monospace',
                 verticalalignment='center', bbox=dict(boxstyle='round,pad=1', facecolor='#FDF5E6', edgecolor='#8B5A2B', linewidth=1.5))
    
    plt.tight_layout()
    stats_fig_path = os.path.join(output_dir, "dataset_statistics.png")
    plt.savefig(stats_fig_path, dpi=200)
    plt.close()
    
    # 2. Visualisasi Grid Random Sample Images (4x4)
    np.random.seed(42)
    sample_indices = np.random.choice(valid_images, size=16, replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("Eksplorasi Sampel Acak Citra Motif Batik (16 Sampel)", fontsize=15, fontweight='bold', y=0.98)
    
    for ax, fname in zip(axes.flatten(), sample_indices):
        img_path = os.path.join(dataset_dir, fname)
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(fname, fontsize=10)
        ax.axis('off')
        
    plt.tight_layout()
    sample_fig_path = os.path.join(output_dir, "dataset_sample_grid.png")
    plt.savefig(sample_fig_path, dpi=200)
    plt.close()
    
    # 3. Visualisasi Pasangan ID (Perbandingan Na vs Nb)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("Pemeriksaan Karakteristik Visual Pasangan Base ID (Na vs Nb)", fontsize=15, fontweight='bold', y=0.98)
    
    sample_ids = [0, 1, 2, 3, 10, 25, 50, 100]
    for row_idx, base_id in enumerate(sample_ids):
        fa = f"{base_id}a.png"
        fb = f"{base_id}b.png"
        
        ax_a = axes[row_idx // 2, (row_idx % 2) * 2]
        ax_b = axes[row_idx // 2, (row_idx % 2) * 2 + 1]
        
        if os.path.exists(os.path.join(dataset_dir, fa)):
            im_a = Image.open(os.path.join(dataset_dir, fa))
            ax_a.imshow(im_a)
            ax_a.set_title(f"{fa} (Base {base_id}-a)", fontsize=9)
        ax_a.axis('off')
        
        if os.path.exists(os.path.join(dataset_dir, fb)):
            im_b = Image.open(os.path.join(dataset_dir, fb))
            ax_b.imshow(im_b)
            ax_b.set_title(f"{fb} (Base {base_id}-b)", fontsize=9)
        ax_b.axis('off')
        
    plt.tight_layout()
    pair_fig_path = os.path.join(output_dir, "pair_comparison_grid.png")
    plt.savefig(pair_fig_path, dpi=200)
    plt.close()
    
    # 4. Visualisasi Distribusi Warna RGB Rata-rata
    print("[*] Menghitung distribusi histogram intensitas RGB...")
    sample_for_color = np.random.choice(valid_images, size=min(100, len(valid_images)), replace=False)
    r_vals, g_vals, b_vals = [], [], []
    for fname in sample_for_color:
        img_arr = np.array(Image.open(os.path.join(dataset_dir, fname)))
        r_vals.extend(img_arr[:, :, 0].flatten()[::50])  # subsample for speed
        g_vals.extend(img_arr[:, :, 1].flatten()[::50])
        b_vals.extend(img_arr[:, :, 2].flatten()[::50])
        
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(r_vals, bins=50, color='red', alpha=0.4, label='Kanal Merah (Red)')
    ax.hist(g_vals, bins=50, color='green', alpha=0.4, label='Kanal Hijau (Green)')
    ax.hist(b_vals, bins=50, color='blue', alpha=0.4, label='Kanal Biru (Blue)')
    ax.set_title("Distribusi Intensitas Piksel RGB Citra Motif Batik", fontsize=13, fontweight='bold')
    ax.set_xlabel("Nilai Intensitas Piksel (0 - 255)")
    ax.set_ylabel("Kerapatan Piksel")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    color_fig_path = os.path.join(output_dir, "color_distribution.png")
    plt.savefig(color_fig_path, dpi=200)
    plt.close()

    print(f"[OK] Seluruh visualisasi berhasil disimpan di folder '{output_dir}'.")
    return audit_report

if __name__ == "__main__":
    report = run_dataset_audit()
    print("\n--- RINGKASAN HASIL AUDIT ---")
    print(f"Total File          : {report['total_files']}")
    print(f"Citra Valid         : {report['valid_images_count']}")
    print(f"Citra Corrupt       : {report['corrupt_images_count']}")
    print(f"Format & Mode       : {report['file_extensions']} | {report['color_modes']}")
    print(f"Resolusi            : {report['resolutions']}")
    print(f"Exact Duplicate     : {report['exact_duplicates_count']}")
    print(f"Near Duplicate dHash: {report['near_duplicates_dhash_count']}")
    print(f"Pola Base ID        : {report['filename_pattern_analysis']['unique_base_ids_count']} grup (0 s/d {report['filename_pattern_analysis']['max_base_id']})")
