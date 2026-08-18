"""
Batik AI Generator — Streamlit Web Application (Improved Model 128x128)
Implementasi Generative Adversarial Network untuk Generasi Citra Motif Batik Menggunakan Dataset Tidak Berlabel.
"""

import io
import os
import sys
import random
import numpy as np
from PIL import Image

import streamlit as st

# Setup Path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for p in [ROOT_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.config import (
        DATASET_STATS,
        EXPERIMENT_RESULTS,
        IMAGE_SIZE,
        LATENT_DIM
    )
    from src.inference import (
        load_generator,
        generate_batik_images,
        create_zip_package,
        get_device
    )
except (ImportError, ModuleNotFoundError):
    from config import (
        DATASET_STATS,
        EXPERIMENT_RESULTS,
        IMAGE_SIZE,
        LATENT_DIM
    )
    from inference import (
        load_generator,
        generate_batik_images,
        create_zip_package,
        get_device
    )

# -----------------------------------------------------------------------------
# Konfigurasi Halaman & Styling Tema Budaya Indonesia
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Batik AI Generator — Generative AI Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    :root {
        --color-primary: #8C5B3F;
        --color-primary-dark: #5A3825;
        --color-accent: #C5A059;
        --color-bg-card: #FDFBF7;
        --color-border: #E8DFD8;
        --color-text-main: #2C221E;
        --color-text-muted: #6E5D53;
    }
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--color-text-main);
    }
    
    h1, h2, h3, .brand-title {
        font-family: 'Cinzel', serif !important;
        letter-spacing: 0.5px;
    }
    
    .brand-header {
        background: linear-gradient(135deg, #3A2312 0%, #5A3825 50%, #8C5B3F 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        color: #FDFBF7;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(58, 35, 18, 0.15);
        border: 1px solid rgba(197, 160, 89, 0.3);
    }
    
    .brand-header h1 {
        color: #FDFBF7;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .brand-header p {
        color: #E8DFD8;
        margin-top: 0.5rem;
        font-size: 1.05rem;
    }
    
    .metric-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(197, 160, 89, 0.2);
        border: 1px solid #C5A059;
        color: #FAF5EE;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.8rem;
    }
    
    .card-surface {
        background-color: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
    }
    
    .batik-img-card {
        background: #FFFFFF;
        border: 1px solid #EAE3DC;
        border-radius: 12px;
        padding: 8px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
    }
    
    .batik-img-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(90, 56, 37, 0.12);
        border-color: var(--color-accent);
    }
    
    .disclaimer-banner {
        background-color: #FAF6F0;
        border-left: 4px solid var(--color-accent);
        padding: 1rem 1.2rem;
        border-radius: 4px 10px 10px 4px;
        font-size: 0.85rem;
        color: #5C4A42;
        margin-top: 2rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Caching Model Generator (Dimuat Sekali Saja)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model Generator...")
def get_cached_generator(model_type: str = "improved_dcgan"):
    """Memuat dan menyimpan model generator di RAM untuk inferensi instan."""
    gen, res = load_generator(model_type=model_type)
    return gen, res

# -----------------------------------------------------------------------------
# Header Utama Aplikasi
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="brand-header">
        <h1>BATIK AI GENERATOR</h1>
        <p>Studio Generasi Citra Motif Batik Sintetis Beresolusi Tinggi Berbasis Deep Convolutional Generative Adversarial Network</p>
        <div style="display: flex; gap: 10px; flex-wrap: wrap;">
            <div class="metric-badge">✨ Resolusi Native: 128 × 128 px</div>
            <div class="metric-badge">🧠 Model: Improved DCGAN & StyleGAN2-ADA</div>
            <div class="metric-badge">📉 FID Terbaik: 2.95 (Held-Out Test)</div>
            <div class="metric-badge">🎨 Dataset: 1.216 Citra Asli</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Sidebar Kontrol & Konfigurasi
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/batik.png", width=64)
    st.markdown("### ⚙️ Pengaturan Studio")
    
    # Navigasi Tab
    menu = st.radio(
        "Pilih Menu:",
        ["🎨 Studio Generasi Batik", "📊 Perbandingan Kualitas Model", "🔬 Informasi Dataset & Audit", "ℹ️ Tentang & Warisan Budaya"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🧠 Pilihan Model GAN")
    model_choice = st.selectbox(
        "Arsitektur Model:",
        [
            "Improved DCGAN 128x128 (Best Model)",
            "StyleGAN2-ADA 128x128",
            "DCGAN Baseline 64x64"
        ],
        index=0
    )
    
    # Pemetaan model internal
    if "Improved DCGAN" in model_choice:
        model_type_key = "improved_dcgan"
    elif "StyleGAN2-ADA" in model_choice:
        model_type_key = "stylegan2_ada"
    else:
        model_type_key = "dcgan_baseline"
        
    st.markdown("---")
    st.markdown("### 🎛️ Parameter Sintesis")
    
    num_images = st.select_slider(
        "Jumlah Citra yang Dihasilkan:",
        options=[4, 8, 12, 16],
        value=8,
        help="Pilih jumlah motif batik yang ingin disintesis dalam satu iterasi."
    )
    
    use_random_seed = st.checkbox("Gunakan Seed Acak (Random)", value=True)
    
    if use_random_seed:
        seed_value = random.randint(1, 999999)
        st.caption(f"🎲 Seed acak aktif: `{seed_value}`")
    else:
        seed_value = st.number_input(
            "Masukkan Seed Manual:",
            min_value=1,
            max_value=9999999,
            value=42,
            step=1,
            help="Seed memungkinkan Anda mereproduksi corak motif batik yang persis sama."
        )
        
    device = get_device()
    st.markdown("---")
    st.caption(f"🖥️ Akselerasi Hardware: **{str(device).upper()}**")

# -----------------------------------------------------------------------------
# TAB 1: STUDIO GENERASI BATIK
# -----------------------------------------------------------------------------
if menu == "🎨 Studio Generasi Batik":
    generator, resolution = get_cached_generator(model_type=model_type_key)
    
    col_ctrl, col_btn = st.columns([3, 1])
    with col_ctrl:
        st.markdown(f"#### 🎨 Generate Motif Batik Sintetis ({resolution} × {resolution} px)")
        st.markdown(f"Model Aktif: **{model_choice}** | Jumlah Output: **{num_images} motif** | Seed: **{seed_value}**")
    with col_btn:
        generate_clicked = st.button("✨ Hasilkan Batik", use_container_width=True, type="primary")
        
    # Session State untuk Menyimpan Hasil Generasi Terakhir
    if "generated_images" not in st.session_state or generate_clicked:
        with st.spinner(f"Menyintesis {num_images} motif batik ({resolution}x{resolution})..."):
            images = generate_batik_images(
                generator=generator,
                num_images=num_images,
                seed=seed_value if not use_random_seed else seed_value,
                latent_dim=LATENT_DIM
            )
            st.session_state.generated_images = images
            st.session_state.current_seed = seed_value
            st.session_state.current_resolution = resolution
            st.session_state.current_model = model_choice
            
    # Tampilkan Galeri Hasil Generasi
    curr_imgs = st.session_state.generated_images
    curr_res = st.session_state.get("current_resolution", resolution)
    
    st.markdown("---")
    st.markdown(f"### 🖼️ Galeri Motif Batik Sintetis ({len(curr_imgs)} Motif — {curr_res}×{curr_res} px)")
    
    # 4 Kolom Responsif
    cols = st.columns(4)
    for idx, img in enumerate(curr_imgs):
        col_idx = idx % 4
        with cols[col_idx]:
            st.markdown(f"<div class='batik-img-card'>", unsafe_allow_html=True)
            st.image(img, use_container_width=True, caption=f"Motif #{idx+1:02d} ({curr_res}x{curr_res})")
            
            # Individual Download Button
            img_buf = io.BytesIO()
            img.save(img_buf, format="PNG")
            st.download_button(
                label=f"💾 Unduh #{idx+1:02d}",
                data=img_buf.getvalue(),
                file_name=f"batik_{curr_res}x{curr_res}_{idx+1:02d}_seed_{st.session_state.current_seed}.png",
                mime="image/png",
                key=f"dl_single_{idx}",
                use_container_width=True
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
    # Batch Download ZIP
    st.markdown("---")
    col_zip1, col_zip2 = st.columns([2, 1])
    with col_zip1:
        st.markdown("##### 📦 Unduh Seluruh Koleksi Hasil Sintesis")
        st.markdown(f"Unduh seluruh {len(curr_imgs)} berkas motif batik dalam satu paket arsip ZIP berkualitas tinggi.")
    with col_zip2:
        zip_buf = create_zip_package(curr_imgs, seed=st.session_state.current_seed, resolution=curr_res)
        st.download_button(
            label=f"📦 Unduh Semua (ZIP — {len(curr_imgs)} Motif)",
            data=zip_buf.getvalue(),
            file_name=f"batik_sintetis_{curr_res}x{curr_res}_seed_{st.session_state.current_seed}.zip",
            mime="application/zip",
            key="dl_all_zip",
            use_container_width=True,
            type="secondary"
        )

# -----------------------------------------------------------------------------
# TAB 2: PERBANDINGAN KUALITAS MODEL
# -----------------------------------------------------------------------------
elif menu == "📊 Perbandingan Kualitas Model":
    st.markdown("### 📊 Evaluasi Komparatif Kualitas Model Generatif")
    st.markdown(
        """
        Untuk membuktikan peningkatan kualitas model secara ilmiah dan terukur, dilakukan evaluasi perbandingan antara 
        **DCGAN Baseline (64×64)**, **Improved DCGAN (128×128)**, dan **StyleGAN2-ADA (128×128)** menggunakan **244 citra held-out test set** yang tidak pernah dilihat saat pelatihan.
        """
    )
    
    # Metrik Cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            """
            <div class="card-surface" style="border-top: 4px solid #C0392B;">
                <h4>🏛️ DCGAN Baseline</h4>
                <p><b>Resolusi:</b> 64 × 64 px</p>
                <p><b>Skor FID:</b> 2020.60</p>
                <p><b>Keragaman L2:</b> 13.94</p>
                <p><b>Status:</b> Baseline (Blurry, Low Detail)</p>
            </div>
            """, unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            """
            <div class="card-surface" style="border-top: 4px solid #27AE60;">
                <h4>🌟 Improved DCGAN 128 (Best)</h4>
                <p><b>Resolusi:</b> 128 × 128 px</p>
                <p><b>Skor FID:</b> 2.95 (Terbaik!)</p>
                <p><b>Keragaman L2:</b> 16.21 (Tertinggi!)</p>
                <p><b>Status:</b> Juara (Tajam, Bebas Mode Collapse)</p>
            </div>
            """, unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            """
            <div class="card-surface" style="border-top: 4px solid #2980B9;">
                <h4>🎨 StyleGAN2-ADA 128</h4>
                <p><b>Resolusi:</b> 128 × 128 px</p>
                <p><b>Skor FID:</b> 3.98</p>
                <p><b>Keragaman L2:</b> 1.01</p>
                <p><b>Status:</b> Candidate (Adaptive Augmentation)</p>
            </div>
            """, unsafe_allow_html=True
        )
        
    # Visual Grid Perbandingan Real vs Baseline vs Improved
    st.markdown("#### 🔍 Grid Perbandingan Visual: Real Held-Out vs Baseline vs StyleGAN2-ADA")
    comp_img_path = "outputs/evaluation/real_vs_baseline_vs_stylegan.png"
    if os.path.exists(comp_img_path):
        st.image(comp_img_path, use_container_width=True, caption="Baris 1: Citra Uji Asli (Held-Out) | Baris 2: DCGAN Baseline 64x64 | Baris 3: Model 128x128")
    else:
        st.info("Grid perbandingan visual tersimpan di outputs/evaluation/.")

# -----------------------------------------------------------------------------
# TAB 3: INFORMASI DATASET & AUDIT
# -----------------------------------------------------------------------------
elif menu == "🔬 Informasi Dataset & Audit":
    st.markdown("### 🔬 Laporan Audit Dataset & Analisis Homogenitas")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Citra Asli", "1.216")
    c2.metric("Citra Corrupt / Rusak", "0")
    c3.metric("Duplikat Biner (MD5)", "0")
    c4.metric("Near-Duplicate (dHash)", "110")
    
    st.markdown("---")
    st.markdown("#### 📈 Visualisasi Distribusi Kecerahan & Keragaman Warna")
    audit_img_path = "outputs/audit/dataset_homogeneity_analysis.png"
    if os.path.exists(audit_img_path):
        st.image(audit_img_path, use_container_width=True, caption="Analisis Karakteristik Visual Dataset Asli 1.216 Citra Motif Batik")
    else:
        st.info("Laporan audit tersimpan di outputs/audit/dataset_homogeneity_report.json")

# -----------------------------------------------------------------------------
# TAB 4: TENTANG & WARISAN BUDAYA
# -----------------------------------------------------------------------------
elif menu == "ℹ️ Tentang & Warisan Budaya":
    st.markdown("### ℹ️ Tentang Batik AI Generator")
    st.markdown(
        """
        **Batik AI Generator** adalah proyek *Generative Artificial Intelligence* yang bertujuan mempelajari representasi visual 
        dari ribuan motif batik tradisional nusantara secara *unsupervised* (tanpa label kelas), kemudian menyintesiskan variasi motif 
        baru yang estetik, orisinal, dan berkualitas tinggi.
        
        #### 🌟 Keunggulan Model Generatif 128×128:
        1. **Resolusi 4x Lebih Padat**: Meningkatkan densitas piksel dari 64×64 (4.096 px) menjadi 128×128 (16.384 px).
        2. **Safe Augmentation (Dihedral $D_4$)**: Menerapkan flip dan rotasi 90°/180°/270° yang sesuai dengan karakteristik simetri batik.
        3. **Evaluasi Objektif & Anti-Leakage**: Pemisahan 80% train dan 20% test berdasarkan Base ID kelompok motif untuk memastikan evaluasi FID valid.
        """
    )

# -----------------------------------------------------------------------------
# Cultural Heritage Disclaimer Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="disclaimer-banner">
        <b>⚠️ Cultural Heritage Disclaimer:</b><br>
        <i>Generated images are AI-generated synthetic patterns and should not be interpreted as historically authentic representations of specific Indonesian batik traditions.</i>
        Motif yang dihasilkan adalah pola sintetis hasil komputasi model AI generatif yang mempelajari distribusi visual dataset, bukan merupakan replika sah atau motif sakral dari daerah tertentu di Nusantara.
    </div>
    """,
    unsafe_allow_html=True
)
