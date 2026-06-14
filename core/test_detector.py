import os
import sys
import cv2
import numpy as np

# Tambahkan project root ke sys.path agar modul core dapat diimpor
project_root = r"c:\Users\vivobook\OneDrive\Documents\TA\personal_image_search"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Definisikan path gambar testing
sample_face_path = r"C:\Users\vivobook\.gemini\antigravity-ide\brain\f4ebb324-8c9d-4bfe-af65-21707e64fea3\sample_face_1780017481211.png"
scratch_dir = r"C:\Users\vivobook\.gemini\antigravity-ide\brain\f4ebb324-8c9d-4bfe-af65-21707e64fea3\scratch"
corrupt_image_path = os.path.join(scratch_dir, "corrupt_image.png")

def test_face_detection_and_alignment(detector_type: str = "mtcnn"):
    print(f"=== Uji Deteksi & Penyelarasan Wajah ({detector_type}) ===")
    from core.detector import FaceDetector
    
    # Inisialisasi detector
    print(f"Menginisialisasi FaceDetector dengan tipe: {detector_type}...")
    detector = FaceDetector(detector_type=detector_type)
    print("Inisialisasi berhasil.")
    
    # Pastikan gambar sampel ada
    print(f"Memeriksa keberadaan gambar sampel: {sample_face_path}")
    assert os.path.exists(sample_face_path), f"Gambar sampel tidak ditemukan di {sample_face_path}"
    
    # Jalankan deteksi
    print("Menjalankan detect_and_crop pada gambar sampel...")
    results = detector.detect_and_crop(sample_face_path)
    
    print(f"Jumlah wajah terdeteksi: {len(results)}")
    
    if len(results) > 0:
        for idx, face_data in enumerate(results):
            print(f"Memvalidasi data wajah terdeteksi ke-{idx + 1}:")
            bbox = face_data["bbox"]
            cropped = face_data["cropped_face"]
            
            print(f"  Bounding Box: {bbox}")
            print(f"  Tipe data cropped: {type(cropped)}")
            print(f"  Dimensi cropped: {cropped.shape}")
            
            assert isinstance(cropped, np.ndarray), "Hasil crop harus berupa numpy array."
            assert cropped.shape == (112, 112, 3), "Dimensi wajah harus tepat (112, 112, 3) dalam format BGR."
            assert len(bbox) == 4, "Bbox harus berupa tuple/list berisi 4 elemen (x, y, w, h)."
            print(f"  Wajah ke-{idx + 1}: VALID")
    else:
        raise AssertionError(f"Detektor {detector_type} gagal mendeteksi wajah pada gambar sampel sehat!")
        
    print(f"Uji Deteksi Wajah Sehat ({detector_type}): SELESAI\n")

def test_robustness_on_corrupt_file(detector_type: str = "mtcnn"):
    print(f"=== Uji Stabilitas / Penanganan File Rusak ({detector_type}) ===")
    from core.detector import FaceDetector
    
    # Pastikan folder scratch ada
    os.makedirs(scratch_dir, exist_ok=True)
    
    # Buat berkas dummy yang corrupt (berisi data non-gambar)
    print(f"Membuat file corrupt di: {corrupt_image_path}")
    with open(corrupt_image_path, "wb") as f:
        f.write(b"CORRUPT BINARY DATA NOT AN IMAGE FILE AT ALL " * 10)
        
    detector = FaceDetector(detector_type=detector_type)
    
    # Jalankan deteksi pada gambar yang rusak
    print("Menjalankan detect_and_crop pada file corrupt...")
    try:
        results = detector.detect_and_crop(corrupt_image_path)
        print(f"Hasil deteksi file corrupt: {results}")
        assert results == [], f"detect_and_crop pada file corrupt ({detector_type}) harus mengembalikan list kosong []"
        print(f"Penanganan file corrupt ({detector_type}): SUKSES (Program tidak crash dan mengembalikan [])")
    except Exception as e:
        raise AssertionError(f"Fungsi detect_and_crop ({detector_type}) crash saat membaca file corrupt: {str(e)}")
        
    # Hapus file corrupt setelah selesai pengujian
    try:
        os.remove(corrupt_image_path)
        print("File corrupt berhasil dibersihkan.")
    except Exception as e:
        print(f"Peringatan: Gagal menghapus file corrupt: {str(e)}")
        
    print(f"Uji Stabilitas File Rusak ({detector_type}): LOLOS\n")

if __name__ == "__main__":
    try:
        # 1. Test MTCNN
        test_face_detection_and_alignment("mtcnn")
        test_robustness_on_corrupt_file("mtcnn")
        
        # 2. Test SCRFD
        test_face_detection_and_alignment("scrfd")
        test_robustness_on_corrupt_file("scrfd")
        
        print("SEMUA UNIT TEST UNTUK KEDUA DETECTOR LOLOS DENGAN SUKSES!")
    except Exception as e:
        print(f"UNIT TEST DETECTOR GAGAL: {str(e)}")
        sys.exit(1)

