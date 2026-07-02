import sys
import numpy as np

# Tambahkan project root ke sys.path agar modul core dapat diimpor
import os
project_root = r"c:\Users\vivobook\OneDrive\Documents\TA\personal_image_search"
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def test_identical_vector_distance():
    print("=== Uji Perhitungan Jarak Vektor Identik ===")
    from core.matcher import FaceMatcher
    
    matcher = FaceMatcher(threshold=0.40)
    
    # Buat vektor acak 512-dimensi
    vec1 = np.random.rand(512).astype(np.float32)
    # L2 normalize secara manual agar magnitude 1.0
    vec1 = vec1 / np.linalg.norm(vec1)
    
    # Hitung jarak antara vektor dengan dirinya sendiri
    distance = matcher.hitung_kesamaan_cosine(vec1, vec1)
    print(f"Jarak Cosine vektor identik: {distance}")
    assert np.isclose(distance, 0.0, atol=1e-6), f"Jarak antara vektor identik harus mendekati 0.0, terhitung: {distance}"
    print("Uji Vektor Identik: LOLOS\n")

def test_vector_search_and_sorting():
    print("=== Uji Penyaringan Threshold & Pengurutan Kemiripan ===")
    from core.matcher import FaceMatcher
    
    matcher = FaceMatcher(threshold=0.40)
    
    # Buat kueri satu arah (one-hot vector)
    query = np.zeros(512, dtype=np.float32)
    query[0] = 1.0
    
    # 1. Wajah identik (similarity = 1.0, distance = 0.0) -> Lolos
    emb1 = np.zeros(512, dtype=np.float32)
    emb1[0] = 1.0
    
    # 2. Wajah mirip (similarity = 0.8, distance = 0.2 <= 0.40) -> Lolos
    emb2 = np.zeros(512, dtype=np.float32)
    emb2[0] = 0.8
    emb2[1] = 0.6 # L2 norm: 0.8^2 + 0.6^2 = 1.0
    
    # 3. Wajah orthogonal / beda jauh (similarity = 0.0, distance = 1.0 > 0.40) -> Gugur
    emb3 = np.zeros(512, dtype=np.float32)
    emb3[1] = 1.0
    
    # Susun cache list dummy
    cache_list = [
        {"id": "face_3_far", "file_name": "foto_far.png", "embedding": emb3, "bbox": (0, 0, 10, 10)},
        {"id": "face_1_identical", "file_name": "foto_identical.png", "embedding": emb1, "bbox": (0, 0, 10, 10)},
        {"id": "face_2_similar", "file_name": "foto_similar.png", "embedding": emb2, "bbox": (0, 0, 10, 10)},
    ]
    
    # Lakukan pencarian
    matches = matcher.cari_foto_cocok(query, cache_list)
    
    print(f"Jumlah kecocokan lolos: {len(matches)} (Diharapkan: 2)")
    assert len(matches) == 2, f"Jumlah hasil saringan salah, terhitung: {len(matches)}"
    
    # Verifikasi pengurutan (terdekat harus pertama)
    print(f"Peringkat 1: {matches[0]['id']} (Jarak Cosine: {matches[0]['cosine_distance']})")
    print(f"Peringkat 2: {matches[1]['id']} (Jarak Cosine: {matches[1]['cosine_distance']})")
    
    assert matches[0]["id"] == "face_1_identical", "Peringkat pertama harus wajah identik."
    assert matches[1]["id"] == "face_2_similar", "Peringkat kedua harus wajah mirip."
    assert matches[0]["cosine_distance"] < matches[1]["cosine_distance"], "Hasil pencarian harus diurutkan menaik berdasarkan jarak Cosine."
    
    print("Uji Penyaringan dan Pengurutan: LOLOS\n")

def test_dedup_multiple_faces_same_photo():
    print("=== Uji Deduplikasi Foto Grup (Beberapa Wajah dalam 1 Foto yang Sama) ===")
    from core.matcher import FaceMatcher

    matcher = FaceMatcher(threshold=0.40)

    query = np.zeros(512, dtype=np.float32)
    query[0] = 1.0

    # Dua wajah berbeda pada FOTO YANG SAMA (foto_grup.png), keduanya cocok dengan kueri
    emb_identical = np.zeros(512, dtype=np.float32)
    emb_identical[0] = 1.0  # distance = 0.0

    emb_similar = np.zeros(512, dtype=np.float32)
    emb_similar[0] = 0.8
    emb_similar[1] = 0.6  # distance = 0.2

    # Satu wajah pada foto lain yang berbeda
    emb_other_photo = np.zeros(512, dtype=np.float32)
    emb_other_photo[0] = 0.9
    emb_other_photo[1] = np.sqrt(1 - 0.9 ** 2)

    cache_list = [
        {"id": "face_A_grup", "file_name": "foto_grup.png", "embedding": emb_similar, "bbox": (0, 0, 10, 10)},
        {"id": "face_B_grup", "file_name": "foto_grup.png", "embedding": emb_identical, "bbox": (50, 50, 10, 10)},
        {"id": "face_C_lain", "file_name": "foto_lain.png", "embedding": emb_other_photo, "bbox": (0, 0, 10, 10)},
    ]

    matches = matcher.cari_foto_cocok(query, cache_list)

    print(f"Jumlah kecocokan setelah deduplikasi: {len(matches)} (Diharapkan: 2 foto unik)")
    assert len(matches) == 2, "Foto yang sama dengan beberapa wajah cocok harus muncul hanya sekali."

    file_names = [m["file_name"] for m in matches]
    assert len(file_names) == len(set(file_names)), "Tidak boleh ada file_name duplikat pada hasil pencocokan."

    # Untuk foto_grup.png, kecocokan yang disimpan harus wajah dengan jarak terkecil (face_B_grup)
    grup_match = next(m for m in matches if m["file_name"] == "foto_grup.png")
    assert grup_match["id"] == "face_B_grup", "Kecocokan terbaik (jarak terkecil) yang harus dipertahankan saat deduplikasi."

    print("Uji Deduplikasi Foto Grup: LOLOS\n")

def test_robustness_on_corrupt_embeddings():
    print("=== Uji Ketahanan Data Corrupt / Embedding Rusak (NF3) ===")
    from core.matcher import FaceMatcher
    
    matcher = FaceMatcher(threshold=0.40)
    query = np.zeros(512, dtype=np.float32)
    query[0] = 1.0
    
    emb_healthy = np.zeros(512, dtype=np.float32)
    emb_healthy[0] = 1.0
    
    # Susun cache list dengan data rusak di dalamnya
    cache_list = [
        {"id": "healthy_face", "file_name": "sehat.png", "embedding": emb_healthy, "bbox": (0, 0, 10, 10)},
        {"id": "none_embedding", "file_name": "corrupt1.png", "embedding": None, "bbox": (0, 0, 10, 10)},
        {"id": "wrong_dim", "file_name": "corrupt2.png", "embedding": np.zeros(100, dtype=np.float32), "bbox": (0, 0, 10, 10)},
    ]
    
    try:
        matches = matcher.cari_foto_cocok(query, cache_list)
        print(f"Hasil deteksi data rusak: Terload {len(matches)} data sehat.")
        assert len(matches) == 1, "Sistem harus sukses memproses dan melewati embedding rusak secara otomatis."
        assert matches[0]["id"] == "healthy_face", "Data terload harus data yang sehat saja."
        print("Uji Ketahanan Data Rusak: LOLOS (System gracefully ignored corrupt data without crashing)\n")
    except Exception as e:
        raise AssertionError(f"Sistem crash saat memproses data cache dengan embedding rusak: {str(e)}")

if __name__ == "__main__":
    try:
        test_identical_vector_distance()
        test_vector_search_and_sorting()
        test_dedup_multiple_faces_same_photo()
        test_robustness_on_corrupt_embeddings()
        print("SEMUA UNIT TEST UNTUK MATCHER LOLOS DENGAN SUKSES!")
    except Exception as e:
        print(f"UNIT TEST MATCHER GAGAL: {str(e)}")
        sys.exit(1)
