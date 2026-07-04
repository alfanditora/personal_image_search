"""
generate_corrupt_test_files.py

Script untuk membuat dataset uji STABILITAS (TA Bab VI.1.4 - Pengujian Stabilitas)
dengan menyisipkan berbagai jenis file foto rusak/corrupt ke dalam folder
dataset pengujian yang sudah ada.

Jenis korupsi yang disimulasikan:
1. File kosong (0 byte) berekstensi gambar
2. File truncated (dipotong ke ukuran byte tetap yang sangat kecil -
   simulasi transfer/simpan terputus, dijamin tidak bisa didekode
   sama sekali terlepas dari resolusi foto aslinya)
3. File berisi byte acak total (header tidak valid sama sekali)
4. File asli valid tapi header/magic bytes-nya diacak (simulasi bit-rot)
5. File teks biasa yang diberi ekstensi gambar (mismatch ekstensi vs isi)

Cara pakai:
    python generate_corrupt_test_files.py
(sesuaikan DATASET_DIR dan CONTOH_GAMBAR_VALID di bagian bawah file)
"""

import os
import random


def buat_file_kosong(output_path):
    """Simulasi 1: file 0 byte."""
    open(output_path, "wb").close()


def buat_file_truncated(source_image_path, output_path, byte_disisakan=2048):
    """Simulasi 2: file dipotong di tengah proses transfer/simpan.

    PENTING: menyisakan byte dalam JUMLAH TETAP (bukan persentase dari
    ukuran file asli). JPEG didekode baris-per-baris dari atas ke bawah,
    dan decoder (cv2/PIL) sangat toleran -- kalau kita potong berdasarkan
    persentase (mis. sisakan 50%), bagian ATAS gambar yang masih utuh
    tetap bisa didekode, dan kalau wajah kebetulan ada di bagian itu,
    wajah tetap terdeteksi (bukan file yang benar-benar corrupt).

    Dengan menyisakan byte dalam jumlah kecil & tetap (default 2 KB),
    hanya header + beberapa baris pertama saja yang tersisa -- tidak
    cukup untuk mendekode area wajah pada foto beresolusi normal,
    berapa pun ukuran resolusi aslinya.
    """
    with open(source_image_path, "rb") as f:
        data = f.read()
    potong_di = min(byte_disisakan, len(data))
    with open(output_path, "wb") as f:
        f.write(data[:potong_di])


def buat_file_truncated_ringan(source_image_path, output_path, persen_potong=0.5):
    """Varian opsional: truncation ringan (persentase), sengaja masih bisa
    menyisakan area gambar yang valid. Berguna kalau kamu ingin
    mendokumentasikan graceful degradation (sistem tetap mendeteksi wajah
    dari sebagian gambar yang valid) sebagai temuan terpisah -- BUKAN
    untuk menguji jalur "file gagal dibaca"."""
    with open(source_image_path, "rb") as f:
        data = f.read()
    potong_di = int(len(data) * persen_potong)
    with open(output_path, "wb") as f:
        f.write(data[:potong_di])


def buat_file_garbage(output_path, ukuran_kb=50):
    """Simulasi 3: file isi byte acak total, header tidak valid sama sekali."""
    with open(output_path, "wb") as f:
        f.write(os.urandom(ukuran_kb * 1024))


def buat_header_rusak(source_image_path, output_path, jumlah_byte_dirusak=20):
    """Simulasi 4: file asli valid, tapi beberapa byte di awal (header/magic
    number) diacak -> mensimulasikan bit-rot pada media penyimpanan."""
    with open(source_image_path, "rb") as f:
        data = bytearray(f.read())
    for i in range(min(jumlah_byte_dirusak, len(data))):
        data[i] = random.randint(0, 255)
    with open(output_path, "wb") as f:
        f.write(data)


def buat_ekstensi_salah(output_path, teks="ini bukan file gambar, hanya teks biasa"):
    """Simulasi 5: file teks biasa yang diberi ekstensi gambar (.jpg/.png)."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(teks)


def sisipkan_file_corrupt(dataset_dir, contoh_gambar_valid, jumlah_per_jenis=5):
    """
    Menyisipkan file-file corrupt ke dalam folder dataset pengujian.

    dataset_dir: folder dataset uji (mis. D:/dataset/testset/test_case)
    contoh_gambar_valid: path ke salah satu foto valid di dataset,
                          dipakai sebagai basis untuk simulasi
                          truncated & header rusak.
    jumlah_per_jenis: berapa banyak file corrupt dibuat untuk tiap jenis
    """
    os.makedirs(dataset_dir, exist_ok=True)

    for i in range(jumlah_per_jenis):
        buat_file_kosong(
            os.path.join(dataset_dir, f"corrupt_kosong_{i}.jpg"))
        buat_file_truncated(
            contoh_gambar_valid,
            os.path.join(dataset_dir, f"corrupt_truncated_{i}.jpg"))
        buat_file_garbage(
            os.path.join(dataset_dir, f"corrupt_garbage_{i}.jpg"))
        buat_header_rusak(
            contoh_gambar_valid,
            os.path.join(dataset_dir, f"corrupt_header_{i}.jpg"))
        buat_ekstensi_salah(
            os.path.join(dataset_dir, f"corrupt_ekstensi_{i}.jpg"))

    total = jumlah_per_jenis * 5
    print(f"Berhasil menyisipkan {total} file corrupt "
          f"({jumlah_per_jenis} per jenis, 5 jenis) ke: {dataset_dir}")
    print("Jenis: kosong, truncated, garbage, header_rusak, ekstensi_salah")


if __name__ == "__main__":
    # --- SESUAIKAN DUA VARIABEL INI ---
    DATASET_DIR = r"D:\dataset\testset\test_case"
    CONTOH_GAMBAR_VALID = r"D:\dataset\testset\test_case\0001.jpg"  # foto valid apa saja
    # -----------------------------------

    sisipkan_file_corrupt(DATASET_DIR, CONTOH_GAMBAR_VALID, jumlah_per_jenis=5)