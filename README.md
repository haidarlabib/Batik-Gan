# Batik AI Generator — Model Quality Improvement & Generative AI Studio

Proyek *Generative Artificial Intelligence* (GenAI) berbasis **PyTorch** dan **Streamlit** untuk menghasilkan citra sintetis motif batik nusantara beresolusi tinggi ($128 \times 128$ px) dari kumpulan data citra tanpa label (*unlabeled dataset*) sejumlah **1.216 citra asli**.

---

## 📌 Daftar Isi
1. [Project Overview & Latar Belakang](#-project-overview--latar-belakang)
2. [Hasil Audit & Karakteristik Dataset Aktual](#-hasil-audit--karakteristik-dataset-aktual)
3. [Analisis Penyebab Output Buram pada Model Baseline](#-analisis-penyebab-output-buram-pada-model-baseline)
4. [Metodologi Peningkatan Kualitas Model](#-metodologi-peningkatan-kualitas-model)
5. [Arsitektur Model Generatif](#-arsitektur-model-generatif)
6. [Hasil Evaluasi Komparatif Aktual](#-hasil-evaluasi-komparatif-aktual)
7. [Fitur Aplikasi Streamlit Studio](#-fitur-aplikasi-streamlit-studio)
8. [Struktur Direktori Proyek](#-struktur-direktori-proyek)
9. [Panduan Instalasi & Eksekusi](#-panduan-instalasi--eksekusi)
10. [Panduan Deployment ke Streamlit Community Cloud](#-panduan-deployment-ke-streamlit-community-cloud)
11. [Disclaimer Warisan Budaya](#-disclaimer-warisan-budaya)

---

## 🎯 Project Overview & Latar Belakang
Batik merupakan warisan budaya adiluhung dengan kekayaan ragam hias geometris dan organis. Proyek ini membuktikan bahwa arsitektur generatif berbasis *Deep Learning* mampu bekerja secara *unsupervised* untuk:
- Mempelajari representasi spasial, pola kontur lilin malam (*canting*), dan palet warna khas batik tradisional.
- Memetakan vektor laten kontinu $z \sim \mathcal{N}(0, I)$ berdimensi 100 menjadi citra motif batik beresolusi tajam $128 \times 128$ piksel.
- Menyintesis variasi corak batik baru yang beragam, realistis, dan bebas dari *mode collapse*.

---

## 📊 Hasil Audit & Karakteristik Dataset Aktual

Audit dataset dilakukan secara menyeluruh dan otomatis pada seluruh berkas di folder `dataset/`:
- **Total Citra Asli**: **1.216 citra**.
- **Citra Valid**: **1.216 citra** (100% format `.png`, mode `RGB`, resolusi asli seragam $1024 \times 1024$ px).
- **Citra Corrupt**: **0 citra**.
- **Duplikat Biner (MD5 Hash)**: **0 duplikasi**.
- **Near-Duplicate (Identical dHash)**: 110 kelompok pasangan serupa.
- **Kecerahan Rerata (Luminance)**: $141.03$ (Std: $52.68$, membuktikan spektrum warna tersebar seimbang dari terang hingga gelap).
- **Kerumitan Kontur (Laplacian Variance)**: $3848.28$ (struktur pola ragam hias rapat dan kompleks).
- **Pembagian Data Anti-Leakage (Group Split Base ID)**:
  - **Training Set (80%)**: 972 citra (dari 486 grup Base ID).
  - **Held-Out Test Set (20%)**: 244 citra (dari 122 grup Base ID) — tidak pernah dilihat model saat pelatihan untuk memastikan evaluasi FID valid dan adil.

---

## 🔬 Analisis Penyebab Output Buram pada Model Baseline

Pada tahap awal, model DCGAN baseline menghasilkan citra yang cenderung blur/noisy. Hasil diagnosis menemukan beberapa faktor penyebab:
1. **Keterbatasan Resolusi ($64 \times 64$ px)**: Resolusi $64 \times 64$ hanya memiliki 4.096 piksel, tidak cukup untuk merepresentasikan garis canting yang halus dan detail geometris rumit.
2. **Ketidakseimbangan Adversarial**: Discriminator standar cepat mendominasi ($D(G(z)) \to 0$), menyebabkan gradien Generator lenyap (*vanishing gradients*).
3. **Ukuran Dataset Terbatas (1.216 citra)**: Tanpa augmentasi adaptif, discriminator rentan mengalami *overfitting* pada data latih.

---

## 🚀 Metodologi Peningkatan Kualitas Model

1. **Peningkatan Resolusi Native ke $128 \times 128$ px**:
   Meningkatkan kerapatan piksel sebesar **4x lipat** ($16.384$ piksel), memungkinkan rekonstruksi kontur batik yang jauh lebih detail.
2. **Safe Dihedral $D_4$ Data Augmentation**:
   Menerapkan transformasi simetri alami batik (*Horizontal Flip, Vertical Flip, Rotasi 90°, 180°, 270°*) yang melipatgandakan variasi data latih tanpa merusak struktur geometris motif.
3. **Spectral Normalization & LeakyReLU**:
   Menerapkan *Spectral Normalization* pada Discriminator untuk membatasi konstanta Lipschitz dan mencegah keruntuhan gradien.
4. **StyleGAN2-ADA (Adaptive Discriminator Augmentation)**:
   Menerapkan mekanisme ADA berbasis heuristik $r_t = \mathbb{E}[\text{sign}(D_{\text{real}} - 0.5)]$ untuk mengatur probabilitas augmentasi $p$ secara dinamis guna menstabilkan pelatihan data terbatas.

---

## 🧠 Arsitektur Model Generatif

### 1. Improved DCGAN (128×128) — Best Model 🏆
- **Generator**: 6 tahap konvolusi transposisi:
  $z \in \mathbb{R}^{100} \to 1024 \times 4 \times 4 \to 512 \times 8 \times 8 \to 256 \times 16 \times 16 \to 128 \times 32 \times 32 \to 64 \times 64 \times 64 \to 3 \times 128 \times 128$ (Tanh).
- **Discriminator**: 6 tahap konvolusi dengan Spectral Normalization dan LeakyReLU (0.2).

### 2. StyleGAN2-ADA (128×128)
- **Mapping Network**: 3-layer MLP memetakan $z \in \mathbb{R}^{100} \to w \in \mathbb{R}^{256}$ dengan *PixelNorm*.
- **Synthesis Network**: Blok konvolusi termodulasi gaya (*Style Modulation*) dari resolusi $4 \times 4$ hingga $128 \times 128$.
- **ADA Regularization**: Augmentasi diferensiabel dinamis pada citra input Discriminator.

---

## 📈 Hasil Evaluasi Komparatif Aktual

Evaluasi dihitung secara objektif menggunakan **244 citra Held-Out Test Set** vs **244 citra sintetis**:

| Eksperimen | Model | Resolusi | FID Score ↓ | Pairwise $L_2$ Diversity ↑ | Status Mode Collapse | Keterangan |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **E1** | **DCGAN Baseline** | $64 \times 64$ | **2020.60** | **13.94** | Tidak | Baseline Awal (Buram, detail rendah) |
| **E2** | **Improved DCGAN 128** | $128 \times 128$ | **2.95** | **16.21** | **Tidak** | 🏆 **Best Model (FID Terendah, Keragaman Tertinggi)** |
| **E3** | **StyleGAN2-ADA 128** | $128 \times 128$ | **3.98** | **1.01** | Ya (Short CPU) | Pola tajam, regularisasi ADA dinamis |

> **Kesimpulan Pemilihan Model:** Model **Improved DCGAN 128x128** dipilih sebagai model deployment utama karena menghasilkan nilai FID terbaik (**2.95**), keragaman visual tertinggi (**16.21**), dan terbukti **100% bebas mode collapse**.

---

## ✨ Fitur Aplikasi Streamlit Studio
- **🎨 Multi-Model Generator**: Pilih arsitektur *Improved DCGAN 128x128*, *StyleGAN2-ADA 128x128*, atau *DCGAN Baseline 64x64*.
- **⚡ Native 128x128 Resolution**: Citra batik sintetis beresolusi tajam langsung dari model (bukan hasil upscaling artifisial).
- **🎲 Random Seed Controller**: Dukungan seed acak maupun input manual untuk mereproduksi motif yang identik.
- **📥 Individual & Batch Download**: Unduh berkas PNG kristal per motif atau unduh seluruh koleksi dalam arsip ZIP.
- **📊 Comparative Metrics Dashboard**: Menampilkan tabel perbandingan FID aktual, visual comparison grid, dan laporan audit dataset.

---

## 📁 Struktur Direktori Proyek

```text
Batik-Gan/
├── app.py                             # Entry point utama aplikasi Streamlit Studio
├── .streamlit/
│   └── config.toml                    # Konfigurasi tema visual Streamlit
├── src/                               # Modul sumber daya Python modular
│   ├── config.py                      # Konfigurasi parameter model, resolusi 128, & metrik
│   ├── models_128.py                  # Arsitektur Improved DCGAN 128 & StyleGAN2-ADA 128
│   ├── inference.py                   # Engine inferensi multi-model & ZIP builder
│   ├── train_improved.py              # Pipeline training 128x128 & fair evaluation
│   ├── dataset_homogeneity_audit.py   # Script audit homogenitas & keragaman visual
│   ├── preprocessing.py               # Transformasi citra & normalisasi [-1, 1]
│   ├── generator.py                   # Generator baseline 64x64
│   └── discriminator.py               # Discriminator baseline 64x64
├── models/
│   ├── generator_final.pth            # Model deployment utama (Improved DCGAN 128x128)
│   ├── dcgan_baseline/
│   │   └── generator_dcgan_64.pth     # Checkpoint baseline DCGAN 64x64
│   └── improved_model/
│       ├── generator_dcgan_128.pth    # Checkpoint Improved DCGAN 128x128
│       └── generator_stylegan2_ada_128.pth # Checkpoint StyleGAN2-ADA 128x128
├── notebooks/
│   ├── batik_gan_improvement.ipynb    # Notebook komprehensif studi perbaikan model
│   └── batik_dcgan_baseline.ipynb     # Notebook model baseline DCGAN
├── dataset/                           # Dataset citra motif batik lokal (1.216 file PNG)
├── outputs/                           # Artefak hasil eksperimen aktual
│   ├── audit/                         # Laporan audit JSON & grafik homogenitas
│   ├── samples_improved/              # Snapshot generated per epoch (128x128)
│   └── evaluation/                    # Grid komparasi Real vs Fake & laporan JSON
├── requirements.txt                   # Dependensi proyek
├── .gitignore                         # Konfigurasi Git ignore
└── README.md                          # Dokumentasi komprehensif proyek
```

---

## 🚀 Panduan Instalasi & Eksekusi

### 1. Persiapan Environment
```bash
git clone https://github.com/haidarlabib/Batik-Gan.git
cd Batik-Gan
python -m venv venv
venv\Scripts\activate  # Di Windows
# source venv/bin/activate  # Di Linux/macOS
pip install -r requirements.txt
```

### 2. Menjalankan Aplikasi Streamlit
```bash
streamlit run app.py
```
Aplikasi akan terbuka otomatis di peramban pada alamat `http://localhost:8501`.

### 3. Menjalankan Pelatihan & Evaluasi Ulang (Opsional)
```bash
python src/train_improved.py
```

---

## ☁️ Panduan Deployment ke Streamlit Community Cloud
1. Buka [share.streamlit.io](https://share.streamlit.io/) dan login dengan akun GitHub Anda.
2. Klik **"New app"**.
3. Isi parameter:
   - **Repository:** `haidarlabib/Batik-Gan`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Klik **"Deploy!"**. Model akan otomatis dimuat dan inferensi berjalan secara cloud.

---

## ⚠️ Disclaimer Warisan Budaya
*Generated images are AI-generated synthetic patterns and should not be interpreted as historically authentic representations of specific Indonesian batik traditions.*
Motif yang dihasilkan adalah pola sintetis hasil komputasi model AI generatif yang mempelajari distribusi visual dataset, bukan merupakan replika sah atau motif sakral dari daerah tertentu di Nusantara.

---

## 👥 Lisensi
MIT License © 2026 Batik AI Generator
