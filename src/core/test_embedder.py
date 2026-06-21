import os
import sys
import numpy as np

# Tambahkan project root ke sys.path agar modul core dapat diimpor
project_root = r"c:\Users\vivobook\OneDrive\Documents\TA\personal_image_search"
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def test_weights_preparation():
    print("=== Uji Penyiapan Weights (prepare_weights) ===")
    from core.embedder import prepare_weights
    
    # Jalankan penyiapan weights
    success = prepare_weights()
    print(f"Hasil prepare_weights(): {success}")
    assert success is True, "prepare_weights() mengembalikan False, penyiapan weights gagal."
    
    # Periksa keberadaan file target
    home_dir = os.path.expanduser("~")
    target_path = os.path.join(home_dir, ".deepface", "weights", "arcface_weights.h5")
    print(f"Memeriksa path target cache: {target_path}")
    assert os.path.exists(target_path), "File bobot model tidak ditemukan di folder cache ~/.deepface/weights/"
    
    size = os.path.getsize(target_path)
    print(f"File bobot siap, ukuran: {size} bytes ({size / (1024*1024):.2f} MB)")
    assert size > 130 * 1024 * 1024, "Ukuran file bobot model tidak sesuai (terlalu kecil)."
    print("Uji Penyiapan Weights: LOLOS\n")

def test_embedding_extraction():
    print("=== Uji Ekstraksi Embedding (ArcFaceEmbedder) ===")
    from core.embedder import ArcFaceEmbedder
    
    # Inisialisasi embedder
    print("Menginisialisasi ArcFaceEmbedder...")
    embedder = ArcFaceEmbedder()
    print("Inisialisasi berhasil.")
    
    # Buat dummy face image (112x112x3 RGB/BGR)
    print("Membuat gambar wajah dummy (112x112 BGR)...")
    dummy_face = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    
    # Ekstraksi embedding
    print("Mengekstrak embedding...")
    embedding = embedder.extract_embedding(dummy_face)
    
    # Verifikasi tipe dan dimensi
    print(f"Tipe data output: {type(embedding)}")
    print(f"Dimensi output: {embedding.shape}")
    assert isinstance(embedding, np.ndarray), "Output harus berupa numpy array."
    assert embedding.shape == (512,), "Dimensi embedding harus (512,)."
    
    # Verifikasi L2 Normalization
    l2_norm = np.linalg.norm(embedding)
    print(f"L2 Norm (Magnitude): {l2_norm}")
    assert np.isclose(l2_norm, 1.0, atol=1e-5), f"Embedding tidak ternormalisasi L2 secara absolut (L2 Norm = {l2_norm})"
    
    print("Uji Ekstraksi Embedding: LOLOS\n")

if __name__ == "__main__":
    try:
        test_weights_preparation()
        test_embedding_extraction()
        print("SEMUA UNIT TEST UNTUK EMBEDDER LOLOS DENGAN SUKSES!")
    except Exception as e:
        print(f"UNIT TEST GAGAL: {str(e)}")
        sys.exit(1)
