import os
import logging
import threading
from pathlib import Path
import cv2
import numpy as np

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineWorker(threading.Thread):
    def __init__(self, folder_path: str, on_progress, on_finished, on_error, detector=None, embedder=None):
        """
        Background worker untuk memproses pemindaian gambar, deteksi wajah,
        penyelarasan afin, ekstraksi embedding ArcFace, dan penulisan cache Zero-DB.
        
        Args:
            folder_path (str): Jalur direktori utama tempat kumpulan foto disimpan.
            on_progress (callable): Callback untuk melaporkan kemajuan (current, total, file_name).
            on_finished (callable): Callback saat proses selesai (jumlah_terproses).
            on_error (callable): Callback jika terjadi galat fatal.
        """
        super().__init__()
        self.folder_path = folder_path
        self.on_progress = on_progress
        self.on_finished = on_finished
        self.on_error = on_error
        self.detector = detector
        self.embedder = embedder
        self._stop_event = threading.Event()
        self.daemon = True # Biarkan thread mati jika aplikasi utama ditutup

    def stop(self):
        """Menghentikan jalannya thread worker secara aman."""
        self._stop_event.set()

    def stopped(self) -> bool:
        """Memeriksa apakah flag berhenti telah diatur."""
        return self._stop_event.is_set()

    def run(self):
        try:
            logger.info("PipelineWorker diluncurkan di background.")
            
            # Import backend secara dinamis di dalam thread agar aman
            from utils.file_manager import FileManager
            from core.detector import FaceDetector
            from core.embedder import ArcFaceEmbedder
            from cache_io.cache_handler import CacheHandler
            
            # 1. Inisialisasi utilitas
            fm = FileManager(self.folder_path)
            detector = self.detector if self.detector is not None else FaceDetector()
            embedder = self.embedder if self.embedder is not None else ArcFaceEmbedder()
            cache_handler = CacheHandler(self.folder_path)
            
            # Pemuatan cache yang ada untuk incremental indexing (menghindari re-proses file yang sudah ada)
            try:
                existing_cache = cache_handler.muat_seluruh_cache()
                processed_files = {entry["file_name"] for entry in existing_cache}
                logger.info(f"Ditemukan {len(processed_files)} file yang sudah terindeks di cache.")
            except Exception as e:
                processed_files = set()
                logger.warning(f"Gagal memuat cache lama untuk pemindaian inkremental: {str(e)}")
            
            # 2. Pindai berkas foto
            if self.stopped():
                return
            photo_list = fm.dapatkan_daftar_foto()
            total_photos = len(photo_list)
            
            if total_photos == 0:
                logger.info("Tidak ada gambar valid yang ditemukan di direktori input.")
                self.on_finished(0)
                return
                
            processed_count = 0
            
            # 3. Proses sekuensial masing-masing foto (batch processing)
            for idx, photo_path in enumerate(photo_list):
                # Proteksi anti-leak dan anti-freeze saat ditutup paksa
                if self.stopped():
                    logger.info("PipelineWorker dihentikan oleh pengguna secara paksa.")
                    return
                    
                file_name_rel = str(photo_path.relative_to(fm.input_dir))
                
                # Inkremental: Skip jika file sudah pernah diproses dan ada di cache
                if file_name_rel in processed_files:
                    processed_count += 1
                    # Kirim kemajuan realtime ke UI utama
                    self.on_progress(idx + 1, total_photos, f"[Cached] {photo_path.name}")
                    continue
                
                try:
                    # Validasi biner & muat citra
                    img = fm.validasi_dan_baca_citra(str(photo_path))
                    if img is None:
                        # Gambar corrupt, abaikan dan lanjut sekuensial (NF3)
                        continue
                        
                    # Deteksi dan crop (menghasilkan wajah 112x112 yang di-align)
                    faces = detector.detect_and_crop(str(photo_path))
                    
                    # Simpan cache metadata masing-masing wajah terdeteksi
                    for face_idx, face_data in enumerate(faces):
                        if self.stopped():
                            return
                        cropped_face = face_data["cropped_face"]
                        bbox = face_data["bbox"]
                        
                        # Ekstraksi embedding 512-dimensi L2 normalized
                        embedding = embedder.extract_embedding(cropped_face)
                        
                        # Simpan ke cache JSON lokal
                        cache_handler.simpan_cache(file_name_rel, embedding, bbox)
                        
                    processed_count += 1
                    
                except Exception as e:
                    logger.warning(f"Modul gagal memproses file {photo_path.name}: {str(e)}. Melanjutkan batch...")
                    
                # Kirim kemajuan realtime ke UI utama
                self.on_progress(idx + 1, total_photos, photo_path.name)
                
            # Jalankan callback selesai
            if not self.stopped():
                self.on_finished(processed_count)
                
        except Exception as e:
            logger.error(f"Kegagalan fatal pada background PipelineWorker: {str(e)}")
            self.on_error(str(e))


class SearchWorker(threading.Thread):
    def __init__(self, folder_path: str, query_img_path: str, threshold: float, on_progress, on_finished, on_error, detector=None, embedder=None):
        """
        Background worker untuk mencocokkan foto selfie kueri dengan seluruh data cache RAM.
        
        Args:
            folder_path (str): Jalur direktori utama tempat kumpulan foto disimpan.
            query_img_path (str): Jalur berkas kueri foto selfie sebagai acuan.
            threshold (float): Ambang batas jarak Cosine Distance.
            on_progress (callable): Callback untuk memperbarui log status.
            on_finished (callable): Callback saat pencarian selesai (matches, output_folder).
            on_error (callable): Callback jika terjadi galat.
        """
        super().__init__()
        self.folder_path = folder_path
        self.query_img_path = query_img_path
        self.threshold = threshold
        self.on_progress = on_progress
        self.on_finished = on_finished
        self.on_error = on_error
        self.detector = detector
        self.embedder = embedder
        self._stop_event = threading.Event()
        self.daemon = True

    def stop(self):
        self._stop_event.set()

    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def run(self):
        try:
            logger.info("SearchWorker diluncurkan di background.")
            
            # Import backend secara dinamis di dalam thread agar aman
            from utils.file_manager import FileManager
            from core.detector import FaceDetector
            from core.embedder import ArcFaceEmbedder
            from core.matcher import FaceMatcher
            from cache_io.cache_handler import CacheHandler
            
            # Inisialisasi
            fm = FileManager(self.folder_path)
            detector = self.detector if self.detector is not None else FaceDetector()
            embedder = self.embedder if self.embedder is not None else ArcFaceEmbedder()
            matcher = FaceMatcher(threshold=self.threshold)
            cache_handler = CacheHandler(self.folder_path)
            
            # 1. Prapemrosesan Foto Selfie Kueri
            if self.stopped():
                return
            self.on_progress("Memproses dan memvalidasi citra kueri foto selfie...")
            
            selfie_img = fm.validasi_dan_baca_citra(self.query_img_path)
            if selfie_img is None:
                self.on_error("Berkas kueri foto selfie rusak, kosong, atau format tidak didukung.")
                return
                
            # Deteksi wajah pada foto selfie kueri
            selfie_faces = detector.detect_and_crop(self.query_img_path)
            if not selfie_faces:
                self.on_error("Tidak ada wajah yang terdeteksi pada foto selfie kueri. Gunakan foto lain yang lebih jelas.")
                return
                
            # Ambil wajah pertama terdeteksi pada kueri
            self_face_data = selfie_faces[0]
            selfie_cropped = self_face_data["cropped_face"]
            
            # Ekstraksi embedding selfie kueri (L2 Normalized)
            selfie_embedding = embedder.extract_embedding(selfie_cropped)
            
            # 2. Muat Basis Data Cache RAM secara Batch
            if self.stopped():
                return
            self.on_progress("Memuat basis data cache wajah lokal (.face_cache/) ke RAM...")
            cache_list = cache_handler.muat_seluruh_cache()
            
            if not cache_list:
                self.on_error("Basis data cache wajah lokal kosong. Silakan lakukan 'Pindai & Indeks Foto' terlebih dahulu.")
                return
                
            # 3. Jalankan Pencocokan Wajah Ter-vektorisasi (Vektor Search)
            if self.stopped():
                return
            self.on_progress("Menganalisis kemiripan wajah menggunakan komputasi ter-vektorisasi ArcFace...")
            matches = matcher.cari_foto_cocok(selfie_embedding, cache_list)
            
            if not matches:
                self.on_progress("Pencocokan selesai. Tidak ada foto yang memiliki kemiripan di bawah threshold.")
                self.on_finished([], "")
                return
                
            # 4. Salin Foto Hasil Pencocokan Bebas Overwrite
            if self.stopped():
                return
            self.on_progress(f"Menyalin {len(matches)} foto hasil kecocokan ke folder 'Hasil_Pencarian_Selfie'...")
            
            match_paths = [item["file_path"] for item in matches]
            output_folder = fm.salin_hasil_cocok(match_paths)
            
            if not self.stopped():
                self.on_progress("Pencarian selesai dengan sukses!")
                self.on_finished(matches, output_folder)
                
        except Exception as e:
            logger.error(f"Kegagalan fatal pada background SearchWorker: {str(e)}")
            self.on_error(str(e))
