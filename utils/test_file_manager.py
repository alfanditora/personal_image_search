import os
import sys
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

# Tambahkan project root ke sys.path agar modul core/utils dapat diimpor
project_root = r"c:\Users\vivobook\OneDrive\Documents\TA\personal_image_search"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Definisikan path folder dummy di scratch
scratch_dir = r"C:\Users\vivobook\.gemini\antigravity-ide\brain\f4ebb324-8c9d-4bfe-af65-21707e64fea3\scratch"
dummy_input_dir = os.path.join(scratch_dir, "file_manager_test_dir")

def test_file_manager_full_flow():
    print("=== Uji Fungsionalitas Lengkap FileManager ===")
    from utils.file_manager import FileManager
    
    # 1. Bersihkan dan siapkan struktur direktori uji dummy
    if os.path.exists(dummy_input_dir):
        shutil.rmtree(dummy_input_dir)
        
    sub1_dir = os.path.join(dummy_input_dir, "sub1")
    sub2_dir = os.path.join(dummy_input_dir, "sub2")
    cache_dir = os.path.join(dummy_input_dir, ".face_cache")
    
    os.makedirs(sub1_dir, exist_ok=True)
    os.makedirs(sub2_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # 2. Buat berkas gambar sehat dummy menggunakan Pillow
    print("Membuat berkas gambar sehat dummy...")
    dummy_img = Image.new("RGB", (100, 100), color="blue")
    
    path_img1 = os.path.join(sub1_dir, "foto_pantai.png")
    path_img2 = os.path.join(sub2_dir, "foto_gunung.jpg")
    path_img3 = os.path.join(dummy_input_dir, "selfie_utama.jpeg")
    
    dummy_img.save(path_img1)
    dummy_img.save(path_img2)
    dummy_img.save(path_img3)
    
    # Buat file gambar di dalam folder cache tersembunyi (harus diabaikan)
    path_cache_img = os.path.join(cache_dir, "cache_temp.png")
    dummy_img.save(path_cache_img)
    
    # 3. Inisialisasi FileManager
    fm = FileManager(dummy_input_dir)
    
    # 4. Uji pemindaian folder dapatkan_daftar_foto
    print("Menguji pemindaian gambar secara rekursif...")
    photo_list = fm.dapatkan_daftar_foto()
    print(f"Berkas yang ditemukan: {[p.name for p in photo_list]}")
    
    # Harus menemukan tepat 3 gambar, dan mengabaikan gambar dalam .face_cache
    assert len(photo_list) == 3, f"Jumlah gambar ditemukan salah: {len(photo_list)}, diharapkan 3."
    assert all(p.name != "cache_temp.png" for p in photo_list), "Berkas di dalam folder .face_cache tidak boleh ter-scan!"
    print("Uji Pemindaian Folder: LOLOS")
    
    # 5. Uji validasi berkas sehat
    print("Menguji pemuatan berkas gambar sehat...")
    img_matrix = fm.validasi_dan_baca_citra(path_img1)
    assert img_matrix is not None, "Gagal memuat gambar sehat."
    assert isinstance(img_matrix, np.ndarray), "Hasil baca harus berupa numpy array."
    print(f"Gambar berhasil dimuat, dimensi: {img_matrix.shape}")
    print("Uji Validasi Berkas Sehat: LOLOS")
    
    # 6. Uji validasi berkas corrupt (NF3)
    print("Menguji penanganan berkas gambar corrupt...")
    path_corrupt = os.path.join(dummy_input_dir, "corrupt_file.jpg")
    with open(path_corrupt, "w") as f:
        f.write("CORRUPT BINARY HEADER NOT AN IMAGE DATA AT ALL")
        
    img_corrupt_matrix = fm.validasi_dan_baca_citra(path_corrupt)
    assert img_corrupt_matrix is None, "Pembacaan file corrupt harus mengembalikan None secara aman."
    print("Penanganan berkas corrupt: SUKSES (Program tidak crash dan mengembalikan None)")
    print("Uji Validasi Berkas Corrupt: LOLOS")
    
    # 7. Uji penyalinan berkas duplikat bebas overwrite (anti-overwrite)
    print("Menguji penyalinan hasil bebas tumpang-tindih (duplikat nama)...")
    # Buat dua gambar berbeda dengan nama SAMA di sub1 dan sub2
    dup_path1 = os.path.join(sub1_dir, "liburan.jpg")
    dup_path2 = os.path.join(sub2_dir, "liburan.jpg")
    
    Image.new("RGB", (50, 50), color="red").save(dup_path1)
    Image.new("RGB", (50, 50), color="green").save(dup_path2)
    
    # Lakukan penyalinan
    output_folder = fm.salin_hasil_cocok([dup_path1, dup_path2])
    print(f"Jalur folder hasil: {output_folder}")
    
    # Cek berkas di folder hasil
    result_files = os.listdir(output_folder)
    print(f"Berkas tersalin di folder hasil: {result_files}")
    
    assert "liburan.jpg" in result_files, "Berkas pertama 'liburan.jpg' tidak ditemukan di folder hasil."
    assert "liburan_1.jpg" in result_files, "Pemberian nama duplikat unik 'liburan_1.jpg' gagal."
    assert len(result_files) == 2, f"Jumlah file tersalin tidak cocok, terhitung: {len(result_files)}"
    print("Uji Penyalinan Bebas Tumpang-tindih: LOLOS\n")

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
        test_file_manager_full_flow()
        cleanup_dummy_directories()
        print("SEMUA UNIT TEST UNTUK FILE MANAGER LOLOS DENGAN SUKSES!")
    except Exception as e:
        print(f"UNIT TEST FILE MANAGER GAGAL: {str(e)}")
        cleanup_dummy_directories()
        sys.exit(1)
