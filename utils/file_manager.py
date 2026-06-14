import os
import shutil
import logging
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileManager:
    def __init__(self, jalur_direktori_input: str):
        """
        Menginisialisasi FileManager untuk direktori utama tempat kumpulan foto disimpan.
        Memvalidasi keberadaan dan keabsahan direktori di sistem operasi.
        
        Args:
            jalur_direktori_input (str): Jalur folder lokal tempat gambar disimpan.
        """
        if not jalur_direktori_input:
            raise ValueError("Jalur direktori input tidak boleh kosong.")
            
        self.input_dir = Path(jalur_direktori_input).resolve()
        
        if not self.input_dir.exists():
            logger.error(f"Direktori input tidak ditemukan di sistem: {self.input_dir}")
            raise FileNotFoundError(f"Direktori tidak ditemukan: {self.input_dir}")
            
        if not self.input_dir.is_dir():
            logger.error(f"Jalur yang dimasukkan bukan sebuah direktori valid: {self.input_dir}")
            raise ValueError(f"Jalur bukan sebuah direktori: {self.input_dir}")
            
        logger.info(f"FileManager siap mengelola direktori: {self.input_dir}")

    def dapatkan_daftar_foto(self) -> list[Path]:
        """
        Membaca direktori input secara sekuensial dan rekursif untuk menemukan seluruh berkas
        foto yang memiliki ekstensi valid (.jpg, .jpeg, .png) secara case-insensitive.
        Secara otomatis mengabaikan berkas di dalam subdirektori cache tersembunyi (misal: .face_cache).
        
        Returns:
            list[Path]: Daftar objek Path berkas gambar yang ditemukan.
        """
        valid_extensions = {".jpg", ".jpeg", ".png"}
        photo_paths = []
        
        try:
            # Melakukan penyusuran rekursif menggunakan rglob
            for path in self.input_dir.rglob("*"):
                # Abaikan berkas di dalam subdirektori tersembunyi (dimulai dengan titik, seperti .face_cache)
                # Dapatkan bagian path relatif terhadap input_dir
                try:
                    rel_path = path.relative_to(self.input_dir)
                    if any(part.startswith(".") for part in rel_path.parts):
                        continue
                except ValueError:
                    continue
                    
                if path.is_file() and path.suffix.lower() in valid_extensions:
                    photo_paths.append(path)
                    
            logger.info(f"Berhasil menemukan {len(photo_paths)} berkas gambar valid (rekursif) di: {self.input_dir}")
            return photo_paths
            
        except Exception as e:
            logger.error(f"Gagal memindai gambar di dalam direktori {self.input_dir}: {str(e)}")
            return []

    def validasi_dan_baca_citra(self, jalur_foto: str) -> np.ndarray or None:
        """
        Memeriksa validitas fisik biner gambar menggunakan Pillow (PIL)
        dan memuat gambarnya menjadi matriks NumPy array menggunakan OpenCV (BGR).
        
        Memenuhi parameter stabilitas NF3: Jika gambar rusak (corrupt) atau kosong,
        fungsi mengembalikan None dan membiarkan batch processing terus berjalan tanpa crash.
        
        Args:
            jalur_foto (str): Jalur absolut berkas gambar yang akan divalidasi.
            
        Returns:
            np.ndarray or None: Matriks gambar BGR jika valid, None jika gambar rusak/tidak didukung.
        """
        if not jalur_foto or not os.path.exists(jalur_foto):
            logger.warning(f"File gambar tidak ditemukan untuk validasi: {jalur_foto}")
            return None
            
        try:
            # 1. Gunakan Pillow untuk validasi struktur biner citra secara ketat (mendeteksi korupsi biner)
            with Image.open(jalur_foto) as img:
                img.verify() # Akan melempar eksepsi jika file corrupt secara struktur
                
            # 2. Muat gambar menggunakan OpenCV setelah dikonfirmasi sehat secara biner
            img_cv = cv2.imread(jalur_foto)
            if img_cv is None:
                logger.warning(f"Matriks gambar kosong atau format tidak didukung oleh OpenCV: {jalur_foto}")
                return None
                
            return img_cv
            
        except Exception as e:
            # Mitigasi crash NF3: catat log analisis galat dan kembalikan None secara aman
            logger.error(f"Berkas gambar corrupt atau rusak secara biner di {jalur_foto}: {str(e)}")
            return None

    def salin_hasil_cocok(self, daftar_jalur_foto: list[str]) -> str:
        """
        Membuat folder baru bernama 'Hasil_Pencarian_Selfie' secara otomatis di dalam direktori input.
        Menyalin (bukan memotong/move) seluruh berkas foto target yang lolos ambang batas kemiripan
        ke dalam folder hasil tersebut dengan penanganan nama duplikat secara aman.
        
        Args:
            daftar_jalur_foto (list[str]): List berisi path absolut berkas foto yang cocok.
            
        Returns:
            str: Path folder hasil tempat berkas disalin.
        """
        output_dir = os.path.join(self.input_dir, "Hasil_Pencarian_Selfie")
        
        try:
            # Buat direktori output secara otomatis
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Folder hasil pencarian siap di: {output_dir}")
        except PermissionError as e:
            logger.error(f"Gagal membuat direktori hasil karena keterbatasan hak akses (PermissionError): {str(e)}")
            return output_dir
        except Exception as e:
            logger.error(f"Terjadi galat tak terduga saat membuat folder hasil: {str(e)}")
            return output_dir
            
        copys_count = 0
        for src_path_str in daftar_jalur_foto:
            src_path = Path(src_path_str)
            if not src_path.exists():
                logger.warning(f"File sumber tidak ditemukan, penyalinan dilewati: {src_path_str}")
                continue
                
            try:
                # Tentukan path tujuan penyalinan
                base_name = src_path.name
                dst_path = os.path.join(output_dir, base_name)
                
                # Penanganan nama duplikat agar tidak saling menimpa (overwrite)
                # Contoh: jika menyalin sub1/foto.jpg dan sub2/foto.jpg, file kedua menjadi foto_1.jpg
                if os.path.exists(dst_path):
                    stem = src_path.stem
                    suffix = src_path.suffix
                    counter = 1
                    while os.path.exists(os.path.join(output_dir, f"{stem}_{counter}{suffix}")):
                        counter += 1
                    dst_path = os.path.join(output_dir, f"{stem}_{counter}{suffix}")
                    
                # Lakukan penyalinan fisik yang mempertahankan metadata berkas
                shutil.copy2(str(src_path), dst_path)
                copys_count += 1
                logger.info(f"Menyalin: {src_path.name} -> {os.path.basename(dst_path)}")
                
            except PermissionError as e:
                logger.error(f"Gagal menyalin {src_path.name} karena folder tujuan bersifat read-only / terproteksi: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Gagal menyalin berkas {src_path.name} ke folder hasil: {str(e)}")
                continue
                
        logger.info(f"Selesai menyalin {copys_count} berkas foto hasil pencocokan ke: {output_dir}")
        return output_dir
