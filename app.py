"""
Batik AI Generator - Streamlit Deployment Application
=====================================================
Aplikasi Generative AI berbasis PyTorch DCGAN untuk menghasilkan motif batik nusantara baru
secara sintetis dari distribusi visual dataset tanpa label (unlabeled image dataset).
"""

import os
import sys
import time
from typing import Optional, List, Tuple
from PIL import Image

import streamlit as st
import torch

# Pastikan root directory berada di sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import (
    APP_TITLE,
    APP_SUBTITLE,
    APP_ICON,
    IMAGE_SIZE,
    NZ,
    NGF,
    NC,
    DEFAULT_NUM_IMAGES,
    AVAILABLE_NUM_IMAGES,
    EVALUATION_METRICS
)
from src.inference import (
    get_device,
    find_checkpoint_path,
    load_generator,
    generate_batik_images,
    create_zip_package
)

# -----------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Batik AI Generator - Generative Art Studio",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS STYLING (Modern Batik Indonesian Aesthetics)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    :root {
        --color-primary: #9C5D27;
        --color-primary-dark: #784217;
        --color-primary-light: #C48B57;
        --color-bg-ivory: #FAF6F0;
        --color-card-bg: #FFFFFF;
        --color-card-border: #EADBCE;
        --color-text-dark: #271A11;
        --color-text-muted: #6E5A4D;
        --color-accent-gold: #D4AF37;
        --color-accent-blue: #1F4E5B;
    }

    /* Main Container Font */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--color-text-dark);
    }

    /* Top Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #2A1A10 0%, #462817 50%, #683D21 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(42, 26, 16, 0.25);
        color: #FFFFFF;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(212, 175, 55, 0.3);
    }

    .hero-container::before {
        content: "";
        position: absolute;
        top: -30%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(212, 175, 55, 0.15) 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .hero-title {
        font-family: 'Cinzel', serif;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: 2px;
        margin: 0;
        color: #F8F3EA;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .hero-title .gold-text {
        color: var(--color-accent-gold);
    }

    .hero-subtitle {
        font-size: 1.05rem;
        font-weight: 400;
        color: #E6D5C3;
        margin-top: 0.5rem;
        margin-bottom: 1.2rem;
        max-width: 820px;
        line-height: 1.6;
    }

    /* Badge Pills */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 1rem;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        background: rgba(255, 255, 255, 0.12);
        color: #F9F3EA;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(4px);
    }

    .badge-pill-gold {
        background: rgba(212, 175, 55, 0.2);
        border-color: rgba(212, 175, 55, 0.5);
        color: #FCE8B2;
    }

    /* Cards */
    .batik-card {
        background: var(--color-card-bg);
        border: 1px solid var(--color-card-border);
        border-radius: 14px;
        padding: 1.4rem;
        box-shadow: 0 4px 12px rgba(42, 26, 16, 0.04);
        margin-bottom: 1.2rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .batik-card:hover {
        box-shadow: 0 6px 18px rgba(42, 26, 16, 0.08);
    }

    .batik-card-header {
        font-family: 'Cinzel', serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--color-primary-dark);
        margin-bottom: 0.6rem;
        border-bottom: 1px solid var(--color-card-border);
        padding-bottom: 0.5rem;
    }

    /* Metric Grid */
    .metric-box {
        background: #FDFBF7;
        border: 1px solid #EAD8C7;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }

    .metric-val {
        font-family: 'Cinzel', serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--color-primary);
        display: block;
    }

    .metric-lbl {
        font-size: 0.8rem;
        color: var(--color-text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    /* Image Showcase Frame */
    .image-card {
        background: #FFFFFF;
        border: 1px solid #E6D7C8;
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        text-align: center;
        transition: all 0.25s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    .image-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 20px rgba(156, 93, 39, 0.15);
        border-color: var(--color-primary-light);
    }

    .image-card img {
        border-radius: 8px;
        width: 100%;
        aspect-ratio: 1 / 1;
        object-fit: cover;
        image-rendering: -webkit-optimize-contrast;
    }

    .image-caption {
        font-weight: 600;
        font-size: 0.85rem;
        color: var(--color-primary-dark);
        margin-top: 8px;
        margin-bottom: 4px;
    }

    /* Disclaimer Box */
    .disclaimer-box {
        background: #F6EFE7;
        border-left: 4px solid var(--color-primary);
        padding: 0.9rem 1.2rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #554032;
        margin-top: 2rem;
        line-height: 1.5;
    }

    /* Custom Primary Button Enhancements */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #A6632B 0%, #7E4519 100%);
        color: #FFFFFF;
        font-weight: 600;
        letter-spacing: 0.5px;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.5rem;
        box-shadow: 0 4px 12px rgba(126, 69, 25, 0.25);
        transition: all 0.2s ease;
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #B97134 0%, #905120 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(126, 69, 25, 0.35);
        color: #FFFDF9;
    }

    /* Download button */
    div.stDownloadButton > button {
        background-color: #FFFFFF;
        color: var(--color-primary-dark);
        border: 1px solid #D8C2AF;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.85rem;
        width: 100%;
        transition: all 0.2s ease;
    }

    div.stDownloadButton > button:hover {
        background-color: #F8F3ED;
        border-color: var(--color-primary);
        color: var(--color-primary);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. MODEL CACHING VIA st.cache_resource
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_generator():
    """
    Memuat model Generator DCGAN satu kali saja saat inisialisasi aplikasi.
    Model disimpan dalam cache memori Streamlit untuk performa inferensi kilat.
    """
    device = get_device()
    ckpt_path = find_checkpoint_path()
    
    if ckpt_path is None:
        return None, "⚠️ File checkpoint model Generator (.pth) tidak ditemukan di folder 'models/' atau 'outputs/checkpoints/'."
        
    try:
        model = load_generator(checkpoint_path=ckpt_path, device=device)
        return model, ckpt_path
    except Exception as e:
        return None, f"⚠️ Gagal memuat model Generator: {str(e)}"

# -----------------------------------------------------------------------------
# 4. INITIALIZE SESSION STATE
# -----------------------------------------------------------------------------
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []  # List of (PIL.Image, bytes)
if "last_seed_used" not in st.session_state:
    st.session_state.last_seed_used = None
if "last_num_generated" not in st.session_state:
    st.session_state.last_num_generated = 0
if "generation_time" not in st.session_state:
    st.session_state.generation_time = 0.0

# -----------------------------------------------------------------------------
# 5. SIDEBAR NAVIGATION & SYSTEM STATUS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 2.5rem; line-height: 1;">🎨</div>
            <h2 style="font-family: 'Cinzel', serif; font-size: 1.3rem; margin: 6px 0 0 0; color: #784217;">BATIK AI STUDIO</h2>
            <p style="font-size: 0.8rem; color: #8C7362; margin-top: 2px;">Generative Art Deployment</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    page = st.radio(
        "Navigasi Halaman:",
        ["🏠 Home", "🎨 Generate Batik", "📊 Model Information", "ℹ️ About & Heritage"],
        index=1  # Default ke Generate Batik
    )
    
    st.markdown("---")
    st.markdown("### 🖥️ Status Sistem")
    
    device_obj = get_device()
    device_name = "NVIDIA CUDA GPU" if device_obj.type == "cuda" else f"CPU Multi-Thread ({os.cpu_count() or 4} cores)"
    st.markdown(f"**Komputasi:** `{device_name}`")
    
    # Pre-check model status
    generator_model, model_info_status = get_cached_generator()
    if generator_model is not None:
        rel_ckpt = os.path.relpath(model_info_status, BASE_DIR) if os.path.exists(model_info_status) else model_info_status
        st.success(f"✓ Model Siap: `{os.path.basename(rel_ckpt)}`")
    else:
        st.warning(model_info_status)
        
    st.markdown("---")
    st.caption("© 2026 Batik AI Generator • PyTorch DCGAN")

# -----------------------------------------------------------------------------
# 6. HERO BANNER (Top of Every Page)
# -----------------------------------------------------------------------------
st.markdown(f"""
    <div class="hero-container">
        <h1 class="hero-title">{APP_ICON} BATIK <span class="gold-text">AI GENERATOR</span></h1>
        <p class="hero-subtitle">
            {APP_SUBTITLE}. Mempelajari distribusi visual dan karakteristik estetika 
            kumpulan motif batik nusantara tanpa label menggunakan arsitektur <strong>Deep Convolutional Generative Adversarial Network (DCGAN)</strong>.
        </p>
        <div class="badge-container">
            <span class="badge-pill badge-pill-gold">⚡ PyTorch DCGAN</span>
            <span class="badge-pill">📐 Resolusi: 64 × 64 px</span>
            <span class="badge-pill">🖼️ 1.216 Dataset Asli</span>
            <span class="badge-pill">🎲 Latent Space: 100-D</span>
            <span class="badge-pill">🛡️ Unsupervised Learning</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. PAGE 1: 🏠 HOME
# -----------------------------------------------------------------------------
if page == "🏠 Home":
    st.markdown("### 🏛️ Selamat Datang di Batik AI Generator Studio")
    
    col_h1, col_h2 = st.columns([3, 2])
    
    with col_h1:
        st.markdown("""
        <div class="batik-card">
            <div class="batik-card-header">Latar Belakang & Motivasi</div>
            <p>
                Batik merupakan warisan budaya dunia takbenda (*Intangible Cultural Heritage*) khas Indonesia yang 
                memiliki ribuan variasi pola geometris, ornamen flora-fauna, dan filosofi mendalam.
            </p>
            <p>
                Proyek ini memanfaatkan kecerdasan buatan generatif (<strong>Generative AI</strong>) untuk 
                merekayasa manifold representasi visual motif batik dari dataset lokal yang <strong>tidak memiliki label kelas</strong>.
                Dengan model <strong>DCGAN</strong>, sistem mampu menyintesis corak batik baru yang unik, estetik, dan belum pernah ada sebelumnya.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="batik-card">
            <div class="batik-card-header">✨ Fitur Utama Aplikasi</div>
            <ul>
                <li><strong>Sintesis Citra Berkecepatan Tinggi:</strong> Menghasilkan 4 hingga 16 motif batik sintetis dalam hitungan detik.</li>
                <li><strong>Kontrol Reproduktibilitas (Random Seed):</strong> Memungkinkan eksplorasi corak acak maupun pengulangan variasi yang disukai.</li>
                <li><strong>Ekspor & Unduh Fleksibel:</strong> Unduh citra individual berformat PNG atau unduh seluruh koleksi dalam paket ZIP.</li>
                <li><strong>Optimasi Memori:</strong> Model hanya dimuat satu kali via <em>Resource Caching</em>, siap berjalan di GPU maupun CPU.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_h2:
        st.markdown("""
        <div class="batik-card">
            <div class="batik-card-header">📊 Ringkasan Dataset & Model</div>
        """, unsafe_allow_html=True)
        
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            st.markdown("""
                <div class="metric-box">
                    <span class="metric-val">1.216</span>
                    <span class="metric-lbl">Total Citra Batik</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            st.markdown("""
                <div class="metric-box">
                    <span class="metric-val">64 × 64</span>
                    <span class="metric-lbl">Resolusi Model</span>
                </div>
            """, unsafe_allow_html=True)
            
        with m_c2:
            st.markdown("""
                <div class="metric-box">
                    <span class="metric-val">100-D</span>
                    <span class="metric-lbl">Dimensi Laten (z)</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            st.markdown("""
                <div class="metric-box">
                    <span class="metric-val">13.94</span>
                    <span class="metric-lbl">Pairwise Diversity</span>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("""
            <div style="margin-top: 1.2rem; text-align: center;">
                <p style="font-size: 0.85rem; color: #7A6658;">
                    Siap mencoba membuat motif batik baru?
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Buka Generator Studio", use_container_width=True):
            st.session_state.nav_page = "🎨 Generate Batik"
            st.rerun()
            
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 8. PAGE 2: 🎨 GENERATE BATIK (Primary Generation Studio)
# -----------------------------------------------------------------------------
elif page == "🎨 Generate Batik":
    st.markdown("### 🎨 Generator Studio: Sintesis Motif Batik")
    st.caption("Pilih jumlah gambar dan konfigurasi seed untuk menyintesis motif batik baru secara real-time.")
    
    # Panel Kontrol
    with st.container():
        st.markdown("""
            <div class="batik-card">
                <div class="batik-card-header">🎛️ Pengaturan Generasi</div>
        """, unsafe_allow_html=True)
        
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
        
        with col_ctrl1:
            num_images = st.select_slider(
                "Jumlah Citra yang Dihasilkan:",
                options=AVAILABLE_NUM_IMAGES,
                value=DEFAULT_NUM_IMAGES,
                help="Pilih jumlah citra batik sintetis yang ingin dibuat."
            )
            
        with col_ctrl2:
            is_random_seed = st.checkbox(
                "🎲 Randomize Seed (Corak Acak)",
                value=True,
                help="Jika dicentang, sistem akan memilih vektor laten secara acak pada setiap generasi."
            )
            
            if not is_random_seed:
                manual_seed = st.number_input(
                    "Masukkan Nilai Seed:",
                    min_value=0,
                    max_value=999999,
                    value=42,
                    step=1,
                    help="Gunakan seed yang sama untuk mereproduksi motif yang identik."
                )
            else:
                manual_seed = None
                
        with col_ctrl3:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            btn_generate = st.button("✨ Generate Batik", use_container_width=True, type="primary")
            
        st.markdown("</div>", unsafe_allow_html=True)

    # Logika Eksekusi Generasi
    if btn_generate:
        if generator_model is None:
            st.error(model_info_status)
        else:
            seed_to_use = None if is_random_seed else int(manual_seed)
            if seed_to_use is None:
                seed_to_use = int(torch.randint(0, 1000000, (1,)).item())
                
            spinner_message = f"✨ Generating {num_images} batik patterns... AI is creating new patterns based on the learned visual distribution of the Batik dataset (Seed: {seed_to_use})."
            
            with st.spinner(spinner_message):
                t_start = time.time()
                try:
                    generated_tuples = generate_batik_images(
                        generator=generator_model,
                        num_images=num_images,
                        seed=seed_to_use,
                        device=get_device()
                    )
                    t_elapsed = time.time() - t_start
                    
                    # Simpan ke session state
                    st.session_state.generated_images = generated_tuples
                    st.session_state.last_seed_used = seed_to_use
                    st.session_state.last_num_generated = num_images
                    st.session_state.generation_time = t_elapsed
                    
                    st.success(f"✓ {num_images} motif batik sintetis berhasil dihasilkan dalam {t_elapsed:.2f} detik! (Seed: `{seed_to_use}`)")
                except Exception as e:
                    st.error(f"⚠️ Gagal menghasilkan motif batik: {str(e)}")

    # Tampilan Galeri Hasil Generasi
    if st.session_state.generated_images:
        st.markdown("---")
        
        # Header Hasil & Tombol Download All
        hdr_col1, hdr_col2 = st.columns([3, 1])
        with hdr_col1:
            st.markdown(f"#### 🖼️ Hasil Generasi Citra Batik ({len(st.session_state.generated_images)} Motif)")
            st.caption(f"Random Seed yang Digunakan: `{st.session_state.last_seed_used}` • Waktu Inferensi: `{st.session_state.generation_time:.2f}s`")
            
        with hdr_col2:
            # Buat ZIP bytes untuk seluruh citra
            zip_data = create_zip_package(
                st.session_state.generated_images,
                prefix=f"batik_seed{st.session_state.last_seed_used}"
            )
            st.download_button(
                label="📦 Download All (ZIP)",
                data=zip_data,
                file_name=f"batik_generated_seed_{st.session_state.last_seed_used}.zip",
                mime="application/zip",
                use_container_width=True
            )
            
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        # Susun Grid Gambar Secara Dinamis (4 Kolom per Baris)
        num_cols = 4
        cols = st.columns(num_cols)
        
        for idx, (pil_img, png_bytes) in enumerate(st.session_state.generated_images):
            col_target = cols[idx % num_cols]
            with col_target:
                with st.container():
                    st.image(
                        pil_img,
                        caption=f"Batik Motif #{idx+1:02d}",
                        use_container_width=True
                    )
                    st.download_button(
                        label=f"📥 Download #{idx+1:02d}",
                        data=png_bytes,
                        file_name=f"batik_motif_{idx+1:02d}_seed_{st.session_state.last_seed_used}.png",
                        mime="image/png",
                        key=f"dl_btn_{idx}",
                        use_container_width=True
                    )
                    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
                    
    else:
        st.info("💡 Klik tombol **'✨ Generate Batik'** di atas untuk mulai menyintesis motif batik baru.")

# -----------------------------------------------------------------------------
# 9. PAGE 3: 📊 MODEL INFORMATION
# -----------------------------------------------------------------------------
elif page == "📊 Model Information":
    st.markdown("### 📊 Informasi Model & Metrik Evaluasi Aktual")
    st.caption("Detail spesifikasi teknis arsitektur jaringan DCGAN dan hasil evaluasi performa model.")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("""
        <div class="batik-card">
            <div class="batik-card-header">🧠 Arsitektur Generator DCGAN</div>
            <table style="width: 100%; font-size: 0.88rem; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #EEE;">
                    <td style="padding: 6px 0; font-weight: 600;">Latent Vector (z):</td>
                    <td style="padding: 6px 0; color: #9C5D27;">100 Dimensi &sim; N(0, I)</td>
                </tr>
                <tr style="border-bottom: 1px solid #EEE;">
                    <td style="padding: 6px 0; font-weight: 600;">Lapisan 1:</td>
                    <td style="padding: 6px 0;">ConvTranspose2d(100 &rarr; 512, k=4, s=1, p=0) + BN + ReLU</td>
                </tr>
                <tr style="border-bottom: 1px solid #EEE;">
                    <td style="padding: 6px 0; font-weight: 600;">Lapisan 2:</td>
                    <td style="padding: 6px 0;">ConvTranspose2d(512 &rarr; 256, k=4, s=2, p=1) + BN + ReLU</td>
                </tr>
                <tr style="border-bottom: 1px solid #EEE;">
                    <td style="padding: 6px 0; font-weight: 600;">Lapisan 3:</td>
                    <td style="padding: 6px 0;">ConvTranspose2d(256 &rarr; 128, k=4, s=2, p=1) + BN + ReLU</td>
                </tr>
                <tr style="border-bottom: 1px solid #EEE;">
                    <td style="padding: 6px 0; font-weight: 600;">Lapisan 4:</td>
                    <td style="padding: 6px 0;">ConvTranspose2d(128 &rarr; 64, k=4, s=2, p=1) + BN + ReLU</td>
                </tr>
                <tr style="border-bottom: 1px solid #EEE;">
                    <td style="padding: 6px 0; font-weight: 600;">Lapisan Output:</td>
                    <td style="padding: 6px 0;">ConvTranspose2d(64 &rarr; 3, k=4, s=2, p=1) + Tanh</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; font-weight: 600;">Resolusi Output:</td>
                    <td style="padding: 6px 0; color: #1F4E5B; font-weight: bold;">64 &times; 64 piksel (RGB, [-1, 1])</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="batik-card">
            <div class="batik-card-header">🛡️ Arsitektur Discriminator DCGAN</div>
            <table style="width: 100%; font-size: 0.88rem; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #EEE;">
                    <td style="padding: 6px 0; font-weight: 600;">Input Image:</td>
                    <td style="padding: 6px 0;">3 &times; 64 &times; 64 Tensor</td>
                </tr>
                <tr style="border-bottom: 1px solid #EEE;">
                    <td style="padding: 6px 0; font-weight: 600;">Konvolusi 1:</td>
                    <td style="padding: 6px 0;">Conv2d(3 &rarr; 64, k=4, s=2, p=1) + LeakyReLU(0.2)</td>
                </tr>
                <tr style="border-bottom: 1px solid #EEE;">
                    <td style="padding: 6px 0; font-weight: 600;">Konvolusi 2:</td>
                    <td style="padding: 6px 0;">Conv2d(64 &rarr; 128, k=4, s=2, p=1) + BN + LeakyReLU(0.2)</td>
                </tr>
                <tr style="border-bottom: 1px solid #EEE;">
                    <td style="padding: 6px 0; font-weight: 600;">Konvolusi 3:</td>
                    <td style="padding: 6px 0;">Conv2d(128 &rarr; 256, k=4, s=2, p=1) + BN + LeakyReLU(0.2)</td>
                </tr>
                <tr style="border-bottom: 1px solid #EEE;">
                    <td style="padding: 6px 0; font-weight: 600;">Konvolusi 4:</td>
                    <td style="padding: 6px 0;">Conv2d(256 &rarr; 512, k=4, s=2, p=1) + BN + LeakyReLU(0.2)</td>
                </tr>
                <tr style="border-bottom: 1px solid #EEE;">
                    <td style="padding: 6px 0; font-weight: 600;">Lapisan Output:</td>
                    <td style="padding: 6px 0;">Conv2d(512 &rarr; 1, k=4, s=1, p=0) + Sigmoid</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; font-weight: 600;">Fungsi Tujuan:</td>
                    <td style="padding: 6px 0; color: #1F4E5B; font-weight: bold;">Probabilitas Real vs Fake [0, 1]</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m2:
        st.markdown("""
        <div class="batik-card">
            <div class="batik-card-header">📈 Metrik Evaluasi Kuantitatif</div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        - **Model Type:** `{EVALUATION_METRICS['model_architecture']}`
        - **Framework:** `{EVALUATION_METRICS['framework']}`
        - **Ukuran Citra Model:** `{EVALUATION_METRICS['image_size']}`
        - **Dimensi Ruang Laten:** `{EVALUATION_METRICS['latent_dimension']}`
        - **Jumlah Dataset Aktual:** `{EVALUATION_METRICS['dataset_total_images']} Citra` (972 Train / 244 Test)
        - **Fréchet Inception Distance (FID):** `{EVALUATION_METRICS['fid_score']:.2f}` *(Baseline 15-Epoch CPU)*
        - **Pairwise L2 Distance (Diversity):** `{EVALUATION_METRICS['pairwise_l2_diversity']:.2f}`
        - **Status Mode Collapse:** `{EVALUATION_METRICS['mode_collapse_status']}`
        """)
        
        st.markdown("""
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="batik-card">
            <div class="batik-card-header">⚙️ Konfigurasi Hyperparameter Pelatihan</div>
            <ul>
                <li><strong>Optimizer:</strong> Adam (&beta;<sub>1</sub> = 0.5, &beta;<sub>2</sub> = 0.999)</li>
                <li><strong>Learning Rate:</strong> 0.0002</li>
                <li><strong>Loss Function:</strong> Binary Cross Entropy Loss (BCELoss)</li>
                <li><strong>Label Smoothing:</strong> Real = 0.9, Fake = 0.0</li>
                <li><strong>In-Memory Caching:</strong> Pre-resized 64x64 tensors (~48 MB di RAM)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 10. PAGE 4: ℹ️ ABOUT & HERITAGE
# -----------------------------------------------------------------------------
elif page == "ℹ️ About & Heritage":
    st.markdown("### ℹ️ Tentang Proyek Batik AI Generator")
    
    st.markdown("""
    <div class="batik-card">
        <div class="batik-card-header">Deskripsi Proyek</div>
        <p>
            <strong>Batik AI Generator</strong> merupakan aplikasi <em>Generative Artificial Intelligence</em> berbasis 
            arsitektur <strong>Deep Convolutional Generative Adversarial Network (DCGAN)</strong> yang dibangun menggunakan 
            framework <strong>PyTorch</strong>.
        </p>
        <p>
            Aplikasi ini dirancang untuk menghasilkan citra motif batik sintetis baru berdasarkan representasi spasial 
            dan palet warna yang dipelajari secara <em>unsupervised</em> dari kumpulan 1.216 citra motif batik nusantara tanpa label.
        </p>
        <p>
            Dalam arsitektur GAN:
            <ul>
                <li><strong>Generator ($G$):</strong> Bertindak sebagai "seniman" yang berusaha menyusun pola visual motif batik dari noise laten $z$.</li>
                <li><strong>Discriminator ($D$):</strong> Bertindak sebagai "kritikus" yang membedakan antara motif batik asli dan sintetis.</li>
                <li><strong>Streamlit Deployment:</strong> Menyediakan antarmuka interaktif yang ringan dan cepat khusus untuk proses <em>inference / generation</em> tanpa melakukan pelatihan ulang.</li>
            </ul>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="batik-card">
        <div class="batik-card-header">Struktur Direktori Proyek</div>
        <pre style="background: #FAF6F0; padding: 10px; border-radius: 8px; font-size: 0.85rem; border: 1px solid #E6D7C8;">
Batik-Gan/
├── app.py                     # Entry point aplikasi Streamlit
├── src/                       # Modul Python
│   ├── config.py              # Konfigurasi & konstanta
│   ├── generator.py           # Arsitektur Generator DCGAN
│   ├── discriminator.py       # Arsitektur Discriminator DCGAN
│   ├── inference.py           # Engine inferensi & download ZIP
│   ├── preprocessing.py       # Normalisasi & denormalisasi
│   ├── dataset.py             # Dataset loader & caching
│   ├── train.py               # Adversarial training pipeline
│   └── evaluate.py            # Evaluasi FID & Diversity
├── models/
│   └── generator_final.pth    # Checkpoint bobot Generator
├── notebooks/
│   └── batik_dcgan.ipynb      # Dokumentasi Jupyter 20 Bab
├── dataset/                   # 1.216 Citra Batik Asli (1024x1024)
├── requirements.txt           # Dependensi Python
└── README.md                  # Dokumentasi komprehensif
        </pre>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 11. POLITE CULTURAL DISCLAIMER (Always Displayed at Bottom)
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Catatan &amp; Disclaimer Warisan Budaya:</strong><br>
        <em>Generated images are AI-generated synthetic patterns and should not be interpreted as historically authentic representations of specific Indonesian batik traditions.</em>
        Motif yang dihasilkan adalah pola sintetis hasil komputasi model AI generatif yang mempelajari distribusi visual dataset, bukan merupakan replika sah atau motif sakral dari daerah tertentu di Nusantara.
    </div>
""", unsafe_allow_html=True)
