"""
Uji integrasi MANUAL untuk DriveManager (bukan bagian dari automated test suite).

Butuh:
  - models/credentials.json (OAuth Desktop app client) sudah ada.
  - Login interaktif via browser pada percobaan pertama (token di-cache ke models/token.json).
  - Sebuah folder Google Drive nyata berisi beberapa foto .jpg/.png untuk diuji.

Cara pakai:
    python src/drive_io/test_drive_manager.py "<tautan_atau_id_folder_drive>"
"""
import os
import sys
import json
import shutil
import numpy as np

project_root = r"c:\Users\vivobook\OneDrive\Documents\TA\personal_image_search"
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

scratch_dir = r"C:\Users\vivobook\AppData\Local\Temp\claude\C--Users-vivobook-OneDrive-Documents-TA-personal-image-search\963b754a-2614-418d-83e3-dcd47e6562bb\scratchpad"
dummy_local_root = os.path.join(scratch_dir, "drive_manager_test_staging")


def test_round_trip(folder_link: str):
    print("=== Uji Round-trip DriveManager (list -> download -> upload cache -> copy hasil) ===")
    from pathlib import Path
    from drive_io.drive_manager import DriveManager

    if os.path.exists(dummy_local_root):
        shutil.rmtree(dummy_local_root)

    print(f"Menghubungkan ke folder Drive: {folder_link}")
    drive = DriveManager(folder_link)

    print("Menyusuri isi folder Drive (list_images)...")
    image_map = drive.list_images()
    print(f"  Ditemukan {len(image_map)} gambar.")
    assert len(image_map) > 0, "Folder uji harus berisi minimal 1 foto .jpg/.png."

    print(f"Mengunduh folder ke staging lokal: {dummy_local_root}")
    local_root = drive.download_folder_to_local(Path(dummy_local_root))
    for rel_path in image_map:
        assert (local_root / rel_path).exists(), f"Berkas {rel_path} gagal diunduh ke staging lokal."
    print("  Semua gambar berhasil diunduh ke staging lokal.")

    print("Menulis satu berkas cache dummy secara lokal...")
    cache_dir = local_root / ".face_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dummy_cache_name = "drivemanager_test_dummy.json"
    dummy_embedding = np.random.rand(512).astype(np.float32).tolist()
    with open(cache_dir / dummy_cache_name, "w", encoding="utf-8") as f:
        json.dump({
            "id": "drivemanager_test_dummy",
            "file_name": next(iter(image_map)),
            "file_path": str(local_root / next(iter(image_map))),
            "bbox": [0, 0, 10, 10],
            "embedding": dummy_embedding,
        }, f)

    print("Mengunggah cache baru ke folder '.face_cache' di Drive...")
    drive.upload_new_cache_files(local_root)

    cache_files_on_drive = drive._list_children(
        drive._cache_folder_id, extra_query="mimeType = 'application/json'"
    )
    names_on_drive = {item["name"] for item in cache_files_on_drive}
    assert dummy_cache_name in names_on_drive, "Berkas cache dummy tidak ditemukan di Drive setelah upload."
    print("  Berkas cache dummy berhasil terverifikasi ada di Drive.")

    print("Menyalin 1 foto ke folder 'Hasil_Pencarian_Selfie' di Drive...")
    sample_rel_path = next(iter(image_map))
    results_folder_id = drive.copy_matches_to_drive_results([sample_rel_path])
    assert results_folder_id, "copy_matches_to_drive_results tidak mengembalikan ID folder hasil."

    result_files = drive._list_children(results_folder_id)
    assert any(f["name"] == os.path.basename(sample_rel_path) for f in result_files), \
        "Foto hasil pencocokan tidak ditemukan di folder 'Hasil_Pencarian_Selfie' Drive."
    print("  Foto hasil pencocokan berhasil terverifikasi ada di Drive.")

    print("Uji Round-trip DriveManager: LOLOS\n")


def cleanup():
    print("=== Membersihkan staging lokal uji ===")
    try:
        if os.path.exists(dummy_local_root):
            shutil.rmtree(dummy_local_root)
            print("Staging lokal uji berhasil dibersihkan.")
    except Exception as e:
        print(f"Peringatan: gagal membersihkan staging lokal: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Pemakaian: python src/drive_io/test_drive_manager.py \"<tautan_atau_id_folder_drive>\"")
        sys.exit(1)

    try:
        test_round_trip(sys.argv[1])
        cleanup()
        print("UJI INTEGRASI DRIVE MANAGER LOLOS DENGAN SUKSES!")
    except Exception as e:
        print(f"UJI INTEGRASI DRIVE MANAGER GAGAL: {str(e)}")
        cleanup()
        sys.exit(1)
