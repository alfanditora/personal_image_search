import os
import sys
import shutil
import numpy as np

# Tambahkan project root ke sys.path agar modul core/cache_io dapat diimpor
project_root = r"c:\Users\vivobook\OneDrive\Documents\TA\personal_image_search"
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Definisikan path folder dummy di scratch
scratch_dir = r"C:\Users\vivobook\.gemini\antigravity-ide\brain\f4ebb324-8c9d-4bfe-af65-21707e64fea3\scratch"
dummy_input_dir = os.path.join(scratch_dir, "dummy_input_dir")

def test_cache_save_and_load():
    print("=== Uji Penyimpanan & Pemuatan Cache Sehat ===")
    from cache_io.cache_handler import CacheHandler
    
    # 1. Bersihkan dan buat folder input terisolasi untuk pengetesan
    if os.path.exists(dummy_input_dir):
        shutil.rmtree(dummy_input_dir)
    os.makedirs(dummy_input_dir, exist_ok=True)
    
    # 2. Inisialisasi CacheHandler
    handler = CacheHandler(dummy_input_dir)
    
    # 3. Buat embedding dummy berdimensi 512
    print("Membuat data embedding wajah dummy (512-dimensi)...")
    original_embedding = np.random.rand(512).astype(np.float32)
    original_bbox = (50, 60, 120, 120)
    original_filename = "selfie_test.png"
    
    # 4. Simpan cache
    print("Menyimpan metadata wajah ke cache lokal...")
    save_success = handler.simpan_cache(original_filename, original_embedding, original_bbox)
    assert save_success is True, "Fungsi simpan_cache() mengembalikan False, penyimpanan gagal."
    print("Penyimpanan cache: SUKSES")
    
    # 5. Muat kembali cache ke RAM
    print("Memuat kembali seluruh cache dari penyimpanan ke RAM...")
    loaded_cache = handler.muat_seluruh_cache()
    
    # 6. Validasi data yang dimuat kembali
    assert len(loaded_cache) == 1, f"Jumlah cache terload tidak sesuai: {len(loaded_cache)}, diharapkan 1."
    
    face_data = loaded_cache[0]
    print(f"Data Wajah Terload:")
    print(f"  ID Wajah: {face_data['id']}")
    print(f"  Nama Berkas Asli: {face_data['file_name']}")
    print(f"  Path Absolut: {face_data['file_path']}")
    print(f"  Bounding Box: {face_data['bbox']}")
    print(f"  Tipe data embedding: {type(face_data['embedding'])}")
    print(f"  Dimensi embedding: {face_data['embedding'].shape}")
    
    # Verifikasi integritas data
    assert face_data["file_name"] == original_filename, "Nama berkas tidak cocok."
    assert tuple(face_data["bbox"]) == original_bbox, "Bbox koordinat tidak cocok."
    assert isinstance(face_data["embedding"], np.ndarray), "Embedding harus dipulihkan ke bentuk numpy array."
    assert face_data["embedding"].shape == (512,), "Dimensi embedding terload harus (512,)."
    
    # Verifikasi kesamaan nilai matriks
    values_match = np.allclose(face_data["embedding"], original_embedding, atol=1e-6)
    print(f"  Nilai matriks identik: {values_match}")
    assert values_match, "Nilai vektor embedding terload berbeda dengan vektor asli yang disimpan."
    
    print("Uji Penyimpanan & Pemuatan Cache Sehat: LOLOS\n")

def test_cache_corruption_handling():
    print("=== Uji Ketahanan Terhadap Cache Rusak / Corrupt (NF3) ===")
    from cache_io.cache_handler import CacheHandler
    
    # Pastikan CacheHandler terhubung
    handler = CacheHandler(dummy_input_dir)
    
    # 1. Tulis berkas cache JSON rusak (corrupt) secara sengaja ke folder cache
    corrupt_cache_path = os.path.join(handler.cache_dir, "corrupt_metadata_face.json")
    print(f"Membuat berkas cache corrupt di: {corrupt_cache_path}")
    with open(corrupt_cache_path, "w", encoding="utf-8") as f:
        f.write("{ invalid_json_structure: true, empty_embedding_field: ") # JSON terpotong/sengaja rusak
        
    # 2. Panggil muat seluruh cache
    print("Memuat seluruh cache (termasuk berkas corrupt)...")
    try:
        loaded_cache = handler.muat_seluruh_cache()
        # Harus tetap sukses terload 1 (dari berkas sehat pada uji pertama),
        # sedangkan berkas corrupt dilewati secara anggun tanpa crash.
        print(f"Jumlah cache terload setelah skip corrupt: {len(loaded_cache)}")
        assert len(loaded_cache) == 1, "Sistem harus melompati file corrupt dan tetap memuat 1 file cache sehat."
        print("Uji Ketahanan Cache Rusak: LOLOS (System successfully skipped corrupt file and loaded healthy file)\n")
    except Exception as e:
        raise AssertionError(f"Sistem crash saat memuat cache yang di dalamnya terdapat berkas corrupt: {str(e)}")

def cleanup_dummy_directories():
    print("=== Membersihkan Folder Uji Dummy ===")
    try:
        if os.path.exists(dummy_input_dir):
            shutil.rmtree(dummy_input_dir)
            print("Folder uji dummy berhasil dibersihkan.")
    except Exception as e:
        print(f"Peringatan: Gagal membersihkan folder dummy: {str(e)}")
    print("Semua proses selesai.\n")

if __name__ == "__main__":
    try:
        test_cache_save_and_load()
        test_cache_corruption_handling()
        cleanup_dummy_directories()
        print("SEMUA UNIT TEST UNTUK CACHE HANDLER LOLOS DENGAN SUKSES!")
    except Exception as e:
        print(f"UNIT TEST CACHE HANDLER GAGAL: {str(e)}")
        cleanup_dummy_directories()
        sys.exit(1)
