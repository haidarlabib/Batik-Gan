"""
Module: flask_app.py
Deskripsi: Backend Flask Server alternatif untuk demo web jika diperlukan.
"""

import os
import sys
import io
import zipfile
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.generate import generate_images

app = Flask(__name__, template_folder="templates", static_folder="static")

GENERATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

@app.route("/")
def index():
    """Halaman utama demo BatikGen."""
    return render_template("index.html")

@app.route("/api/status")
def api_status():
    """Endpoint status untuk memverifikasi kesiapan model."""
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "generator_final.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "checkpoints", "generator_final.pth")
    return jsonify({
        "status": True,
        "model_loaded": os.path.exists(ckpt_path),
        "checkpoint": ckpt_path if os.path.exists(ckpt_path) else None,
        "dataset_size": 1216,
        "image_size": 64
    })

@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Endpoint API untuk menghasilkan N citra motif batik baru."""
    try:
        data = request.get_json() or {}
        num_images = int(data.get("num_images", 16))
        num_images = max(1, min(64, num_images))
        seed = data.get("seed")
        if seed is not None and str(seed).strip() != "":
            seed = int(seed)
        else:
            seed = None
            
        saved_paths = generate_images(
            num_images=num_images,
            output_dir=GENERATED_DIR,
            seed=seed
        )
        
        image_urls = [f"/outputs/generated/{os.path.basename(p)}" for p in saved_paths]
        
        return jsonify({
            "status": "success",
            "count": len(image_urls),
            "seed": seed,
            "images": image_urls
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/outputs/generated/<path:filename>")
def serve_generated(filename):
    """Menyajikan file citra yang baru di-generate."""
    return send_from_directory(GENERATED_DIR, filename)

@app.route("/api/download-all")
def download_all():
    """Mengemas seluruh citra yang ada di folder outputs/generated menjadi file ZIP."""
    files = [f for f in os.listdir(GENERATED_DIR) if f.lower().endswith(".png")]
    if not files:
        return "Belum ada citra yang di-generate", 404
        
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in files:
            fpath = os.path.join(GENERATED_DIR, f)
            zipf.write(fpath, arcname=f)
            
    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name='batik_sintetis_dcgan.zip'
    )

if __name__ == "__main__":
    print("[*] Menjalankan Server Web Demo BatikGen di http://localhost:5000...")
    app.run(host="0.0.0.0", port=5000, debug=False)
