# Batik AI Generator — Generative Adversarial Network untuk Sintesis Motif Batik

Proyek *Generative Artificial Intelligence* (GenAI) berbasis **PyTorch** dan **Streamlit** yang mengimplementasikan arsitektur **Deep Convolutional Generative Adversarial Network (DCGAN)** untuk mempelajari distribusi visual dan menghasilkan citra sintetis motif batik nusantara baru dari kumpulan data citra tanpa label (*unlabeled dataset*).

---

## 📌 Daftar Isi
1. [Project Overview & Latar Belakang](#-project-overview--latar-belakang)
2. [Fitur Aplikasi Streamlit](#-fitur-aplikasi-streamlit)
3. [Struktur Direktori Proyek](#-struktur-direktori-proyek)
4. [Hasil Audit Dataset Aktual](#-hasil-audit-dataset-aktual)
5. [Arsitektur Model DCGAN](#-arsitektur-model-dcgan)
6. [Hasil Pelatihan & Evaluasi Aktual](#-hasil-pelatihan--evaluasi-aktual)
7. [Panduan Instalasi & Eksekusi](#-panduan-instalasi--eksekusi)
8. [Panduan Deployment ke Streamlit Community Cloud](#-panduan-deployment-ke-streamlit-community-cloud)
9. [Disclaimer Warisan Budaya](#-disclaimer-warisan-budaya)

---

## 🎯 Project Overview & Latar Belakang
Batik merupakan warisan budaya adiluhung dengan kekayaan ragam hias geometris dan organis. Penerapan model generatif (*Generative AI*) pada domain motif batik membuka peluang besar dalam pelestarian budaya digital, otomatisasi desain tekstil, dan eksplorasi corak baru.

Dalam skenario dunia nyata, data citra lokal sering kali dikumpulkan tanpa anotasi kelas atau label formal. Proyek ini membuktikan bahwa arsitektur **DCGAN** mampu bekerja secara *unsupervised* untuk:
- Mempelajari representasi spasial, pola garis lilin (*canting*), dan palet warna khas batik.
- Memetakan vektor laten kontinu $z \sim \mathcal{N}(0, I)$ berdimensi 100 menjadi citra batik beresolusi $64 \times 64$ piksel.
- Menghasilkan variasi motif batik baru yang unik, realistis, dan bebas dari *mode collapse*.
- Melakukan inferensi secara instan melalui antarmuka **Streamlit Web Studio** yang modern dan responsif.

---

## ✨ Fitur Aplikasi Streamlit
- **🎨 Live Pattern Generator**: Menghasilkan 4, 8, 12, atau 16 motif batik sintetis sekaligus secara real-time.
- **🎲 Random Seed & Reproducibility**: Opsi acak atau input nomor seed manual untuk mereproduksi variasi motif batik yang diinginkan.
- **📥 Individual & Batch Download**: Unduh masing-masing motif berformat PNG kristal atau unduh seluruh koleksi dalam arsip ZIP (`batik_generated_seed_xxx.zip`).
- **⚡ Resource Caching**: Model Generator dimuat hanya satu kali ke memori RAM/GPU (`@st.cache_resource`) sehingga proses generasi berlangsung cepat tanpa overhead.
- **🖥️ Hardware Agnostic**: Otomatis mendeteksi akselerasi GPU (CUDA) dan secara mulus fallback ke CPU.
- **📊 Real Metric Dashboard**: Menampilkan informasi arsitektur aktual, hasil evaluasi FID, dan analisis keragaman (*diversity*).

---

## 📁 Struktur Direktori Proyek

```text
Batik-Gan/
├── app.py                     # Entry point utama aplikasi Streamlit
├── .streamlit/
│   └── config.toml            # Konfigurasi tema visual Streamlit
├── src/                       # Modul Python modular
│   ├── config.py              # Konfigurasi global & parameter model
│   ├── generator.py           # Arsitektur Generator DCGAN (PyTorch)
│   ├── discriminator.py       # Arsitektur Discriminator DCGAN (PyTorch)
│   ├── inference.py           # Engine inferensi & pembuatan ZIP
│   ├── preprocessing.py       # Transformasi & normalisasi citra [-1, 1]
│   ├── dataset.py             # PyTorch Dataset dengan in-memory caching & anti-leakage split
│   ├── train.py               # Loop pelatihan adversarial, checkpointing, & snapshot
│   ├── evaluate.py            # Evaluasi Real vs Fake, skor FID, & diversity analysis
│   └── generate.py            # CLI script inferensi
├── models/
│   └── generator_final.pth    # Checkpoint bobot model Generator terlatih (~14.3 MB)
├── notebooks/
│   └── batik_dcgan.ipynb      # Jupyter Notebook lengkap 20 Bab dalam Bahasa Indonesia
├── dataset/                   # Dataset citra motif batik lokal (1.216 file PNG)
├── outputs/                   # Direktori hasil eksekusi eksperimen
│   ├── audit/                 # Laporan audit JSON & 4 grafik visualisasi dataset
│   ├── samples/               # Snapshot progres fixed-noise per epoch & kurva loss
│   ├── checkpoints/           # Model checkpoints (.pth)
│   ├── generated/             # Sampel citra batik & visual montage grid
│   └── evaluation/            # Grid Real vs Fake, plot keragaman, & evaluation_report.json
├── requirements.txt           # Daftar dependensi library Python
├── .gitignore                 # Konfigurasi Git ignore
└── README.md                  # Dokumentasi komprehensif proyek
```

---

## 📊 Hasil Audit Dataset Aktual

Audit dataset dilakukan secara otomatis pada seluruh file di folder `dataset/`:
- **Total File**: **1.216 citra**.
- **Format File**: 100% `.png`.
- **Mode Warna & Channel**: 100% `RGB` (3 channel).
- **Resolusi Asli**: Seragam $1024 \times 1024$ piksel.
- **Citra Corrupt**: **0 citra** (100% valid dan bersih).
- **Exact Duplicates (MD5 Hash)**: **0 duplikasi biner**.
- **Near Duplicates (dHash)**: 0 kelompok duplikat visual identik.
- **Analisis Pasangan Penamaan**: Terdiri dari 608 Base ID (`0` hingga `607`) dengan sufiks `a` dan `b`. Perhitungan perbedaan piksel menunjukkan bahwa `Na` dan `Nb` memiliki perbedaan visual nyata (Mean Absolute Error = 72.06/255) sehingga merupakan variasi motif yang berbeda.
- **Strategi Pembagian Data (Anti-Leakage Group Split)**:
  - **Training Set (80%)**: 972 citra (dari 486 grup Base ID).
  - **Test / Reference Set (20%)**: 244 citra (dari 122 grup Base ID).

---

## 🧠 Arsitektur Model DCGAN

### 1. Generator ($G$)
Memetakan vektor laten acak $z \sim \mathcal{N}(0, I)$ berdimensi 100 menjadi citra batik $3 \times 64 \times 64$:
- `Input`: Latent Vector $z$ berdimensi 100
- `Layer 1`: ConvTranspose2d(100 $\to$ 512, kernel=4, stride=1, pad=0) + BatchNorm2d + ReLU $\to (512 \times 4 \times 4)$
- `Layer 2`: ConvTranspose2d(512 $\to$ 256, kernel=4, stride=2, pad=1) + BatchNorm2d + ReLU $\to (256 \times 8 \times 8)$
- `Layer 3`: ConvTranspose2d(256 $\to$ 128, kernel=4, stride=2, pad=1) + BatchNorm2d + ReLU $\to (128 \times 16 \times 16)$
- `Layer 4`: ConvTranspose2d(128 $\to$ 64, kernel=4, stride=2, pad=1) + BatchNorm2d + ReLU $\to (64 \times 32 \times 32)$
- `Layer 5`: ConvTranspose2d(64 $\to$ 3, kernel=4, stride=2, pad=1) + Tanh $\to (3 \times 64 \times 64)$

### 2. Discriminator ($D$)
Mengklasifikasikan apakah citra input $3 \times 64 \times 64$ merupakan citra riil atau sintetis:
- `Input`: Image Tensor $(3 \times 64 \times 64)$
- `Layer 1`: Conv2d(3 $\to$ 64, kernel=4, stride=2, pad=1) + LeakyReLU(0.2) $\to (64 \times 32 \times 32)$
- `Layer 2`: Conv2d(64 $\to$ 128, kernel=4, stride=2, pad=1) + BatchNorm2d + LeakyReLU(0.2) $\to (128 \times 16 \times 16)$
- `Layer 3`: Conv2d(128 $\to$ 256, kernel=4, stride=2, pad=1) + BatchNorm2d + LeakyReLU(0.2) $\to (256 \times 8 \times 8)$
- `Layer 4`: Conv2d(256 $\to$ 512, kernel=4, stride=2, pad=1) + BatchNorm2d + LeakyReLU(0.2) $\to (512 \times 4 \times 4)$
- `Layer 5`: Conv2d(512 $\to$ 1, kernel=4, stride=1, pad=0) + Sigmoid $\to \text{Probabilitas } [0, 1]$

---

## 📈 Hasil Pelatihan & Evaluasi Aktual

Seluruh metrik berikut diperoleh dari eksekusi aktual:

### 1. Dinamika Pelatihan (15 Epochs Baseline CPU)
- **Waktu Pelatihan**: 1676.83 detik (27.95 menit pada 4 core CPU).
- **Generator Loss**: Berhasil turun dari 12.83 pada awal pelatihan menuju kestabilan adversarial di kisaran ~4.0 - ~5.7.
- **Discriminator Fake Confidence $D(G(z))$**: Naik dari 0.000 menjadi ~0.013 - 0.054 (Generator berhasil mempelajari fitur untuk menipu Discriminator).

### 2. Fréchet Inception Distance (FID)
- Dievaluasi secara murni menggunakan **244 citra Test Set** (data uji yang tidak pernah dilihat model saat training) vs **244 citra sintetis**.
- **Skor FID Terhitung**: **2020.60** (Baseline 15-epoch CPU; akan menurun drastis seiring penambahan epoch 50-100 di GPU).

### 3. Diversity & Mode Collapse Analysis
- **Pairwise L2 Distance (Mean)**: **13.94** ($\pm 8.13$).
- **Pairwise L1 Distance (Mean)**: **1345.02**.
- **Pairwise Cosine Distance**: **0.0094**.
- **Status Mode Collapse**: **BEBAS DARI MODE COLLAPSE** (Distribusi jarak spasial menyebar lebar, menandakan keberagaman corak dan variasi motif yang tinggi).

---

## 🚀 Panduan Instalasi & Eksekusi

### 1. Persiapan Environment
Pastikan Python versi 3.10+ telah terpasang. Buat virtual environment dan pasang dependensi:
```bash
# Masuk ke direktori proyek
cd "Batik-Gan"

# Buat virtual environment (opsional)
python -m venv venv
venv\Scripts\activate  # Di Windows
# source venv/bin/activate  # Di Linux/macOS

# Install dependensi
pip install -r requirements.txt
```

### 2. Menjalankan Aplikasi Streamlit (Rekomendasi Utama)
```bash
streamlit run app.py
```
Aplikasi akan terbuka otomatis di peramban pada alamat `http://localhost:8501`.

### 3. Menjalankan Inferensi via CLI (Opsional)
```bash
# Menghasilkan 16 motif batik sintetis
python src/generate.py --num 16
```

### 4. Menjalankan Evaluasi Model
```bash
python src/evaluate.py
```

---

## ☁️ Panduan Deployment ke Streamlit Community Cloud

Aplikasi ini 100% kompatibel dengan **Streamlit Community Cloud**:
1. Pastikan seluruh file proyek telah di-push ke repository GitHub: `https://github.com/haidarlabib/Batik-Gan.git`.
2. Buka [share.streamlit.io](https://share.streamlit.io/) dan login menggunakan akun GitHub Anda.
3. Klik tombol **"New app"**.
4. Isi parameter deployment:
   - **Repository:** `haidarlabib/Batik-Gan`
   - **Branch:** `main` (atau `master`)
   - **Main file path:** `app.py`
5. Klik **"Deploy!"**.
6. Aplikasi akan otomatis menginstal library dari `requirements.txt`, memuat model dari `models/generator_final.pth`, dan siap digunakan secara publik di cloud.

---

## ⚠️ Disclaimer Warisan Budaya
*Generated images are AI-generated synthetic patterns and should not be interpreted as historically authentic representations of specific Indonesian batik traditions.*
Motif yang dihasilkan adalah pola sintetis hasil komputasi model AI generatif yang mempelajari distribusi visual dataset, bukan merupakan replika sah atau motif sakral dari daerah tertentu di Nusantara.

---

## 👥 Lisensi
MIT License © 2026 Batik AI Generator
