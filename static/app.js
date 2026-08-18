/**
 * BatikGen - Interactive Web Demo JavaScript
 */

document.addEventListener('DOMContentLoaded', () => {
    const numSlider = document.getElementById('num-images');
    const numDisplay = document.getElementById('num-display');
    const seedInput = document.getElementById('seed-input');
    const btnRandomSeed = document.getElementById('btn-random-seed');
    const btnGenerate = document.getElementById('btn-generate');
    const btnText = document.getElementById('btn-text');
    const statusDesc = document.getElementById('status-desc');
    const loader = document.getElementById('loader');
    const imageGrid = document.getElementById('image-grid');
    const actionTools = document.getElementById('action-tools');
    const btnDownloadAll = document.getElementById('btn-download-all');

    // Modal elements
    const modal = document.getElementById('lightbox-modal');
    const modalImg = document.getElementById('modal-img');
    const modalCaption = document.getElementById('modal-caption');
    const modalDownload = document.getElementById('modal-download');
    const modalClose = document.getElementById('modal-close');
    const modalBackdrop = document.getElementById('modal-backdrop');

    let currentImages = [];

    // 1. Slider Update
    numSlider.addEventListener('input', (e) => {
        numDisplay.textContent = `${e.target.value} Citra`;
    });

    // 2. Random Seed Button
    btnRandomSeed.addEventListener('click', () => {
        const rSeed = Math.floor(Math.random() * 999999) + 1;
        seedInput.value = rSeed;
    });

    // 3. Generate Request
    btnGenerate.addEventListener('click', async () => {
        const count = parseInt(numSlider.value, 10);
        const seedVal = seedInput.value ? parseInt(seedInput.value, 10) : null;

        // UI State: Loading
        btnGenerate.disabled = true;
        btnText.textContent = "Menyintesis...";
        loader.classList.remove('hidden');
        imageGrid.innerHTML = '';
        actionTools.classList.add('hidden');
        statusDesc.textContent = `Menghasilkan ${count} motif batik sintetis baru...`;

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ num_images: count, seed: seedVal })
            });

            const data = await response.json();

            if (data.status === 'success') {
                currentImages = data.images;
                renderGallery(data.images, data.seed);
                statusDesc.textContent = `Berhasil menghasilkan ${data.images.length} citra motif batik (Seed: ${data.seed || 'Acak'}).`;
                actionTools.classList.remove('hidden');
            } else {
                statusDesc.textContent = `Gagal: ${data.message || 'Terjadi kesalahan sistem'}`;
            }
        } catch (error) {
            console.error(error);
            statusDesc.textContent = "Terjadi kesalahan saat menghubungi server backend.";
        } finally {
            btnGenerate.disabled = false;
            btnText.textContent = "Generate Motif Batik";
            loader.classList.add('hidden');
        }
    });

    // 4. Render Gallery
    function renderGallery(images, seed) {
        imageGrid.innerHTML = '';
        images.forEach((imgUrl, idx) => {
            const card = document.createElement('div');
            card.className = 'batik-card';

            const name = `Batik #${String(idx + 1).padStart(2, '0')}`;

            card.innerHTML = `
                <div class="batik-img-wrapper">
                    <img src="${imgUrl}?t=${Date.now()}" alt="${name}" class="batik-img" loading="lazy">
                </div>
                <div class="batik-info">
                    <span class="batik-tag">${name}</span>
                    <button class="btn-download-mini" title="Download PNG" onclick="event.stopPropagation(); window.open('${imgUrl}', '_blank')">💾</button>
                </div>
            `;

            card.addEventListener('click', () => {
                openLightbox(imgUrl, name);
            });

            imageGrid.appendChild(card);
        });
    }

    // 5. Lightbox Modal
    function openLightbox(url, caption) {
        modalImg.src = url;
        modalCaption.textContent = caption;
        modalDownload.href = url;
        modalDownload.setAttribute('download', `${caption.toLowerCase().replace(/[^a-z0-9]/g, '_')}.png`);
        modal.classList.remove('hidden');
    }

    function closeModal() {
        modal.classList.add('hidden');
    }

    modalClose.addEventListener('click', closeModal);
    modalBackdrop.addEventListener('click', closeModal);

    // 6. Download All as ZIP
    btnDownloadAll.addEventListener('click', () => {
        window.location.href = '/api/download-all';
    });
});
