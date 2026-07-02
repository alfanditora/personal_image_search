import os
import time
import logging
import threading
from pathlib import Path
import cv2
import numpy as np

CHUNK_SIZE = 16  # images per batch-embedding call

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineWorker(threading.Thread):
    def __init__(self, folder_path: str, on_progress, on_finished, on_error, detector=None, embedder=None, drive_link: str = None):
        """
        Background worker untuk memproses pemindaian gambar, deteksi wajah,
        penyelarasan afin, ekstraksi embedding ArcFace, dan penulisan cache Zero-DB.

        Args:
            folder_path (str): Jalur direktori utama tempat kumpulan foto disimpan (mode lokal).
            on_progress (callable): Callback untuk melaporkan kemajuan (current, total, file_name).
            on_finished (callable): Callback saat proses selesai (jumlah_terproses).
            on_error (callable): Callback jika terjadi galat fatal.
            drive_link (str, optional): Tautan/ID folder Google Drive. Jika diisi, folder_path
                diabaikan dan pemrosesan berjalan terhadap folder Drive tersebut (lihat drive_io.DriveManager).
        """
        super().__init__()
        self.folder_path = folder_path
        self.on_progress = on_progress
        self.on_finished = on_finished
        self.on_error = on_error
        self.detector = detector
        self.embedder = embedder
        self.drive_link = drive_link
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

            # 0. Jika sumber input adalah Google Drive, sinkronkan dahulu ke folder staging lokal
            drive = None
            effective_folder = self.folder_path
            drive_download_time = 0.0
            drive_upload_time = 0.0
            if self.drive_link:
                from drive_io.drive_manager import DriveManager
                self.on_progress(0, 1, "Menghubungkan ke Google Drive...")
                drive = DriveManager(self.drive_link)
                local_root = Path.home() / ".personal_image_search" / "drive_cache" / drive.root_folder_id
                self.on_progress(0, 1, "Menyinkronkan foto & cache dari Google Drive...")
                _t0 = time.time()
                drive.download_folder_to_local(local_root)
                drive_download_time = time.time() - _t0
                effective_folder = str(local_root)

            # 1. Inisialisasi utilitas
            fm = FileManager(effective_folder)
            detector = self.detector if self.detector is not None else FaceDetector("scrfd")
            embedder = self.embedder if self.embedder is not None else ArcFaceEmbedder()
            cache_handler = CacheHandler(effective_folder)

            # Reset timing accumulators for this indexing run
            detector.reset_timings()
            embedder.reset_timings()

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
                self.on_finished(0, {
                    "drive_sync_time": drive_download_time + drive_upload_time,
                    "faces_detected": len(existing_cache),
                })
                return

            processed_count = 0
            new_faces_count = 0  # jumlah wajah baru yang berhasil dideteksi pada run ini

            # 3a. Phase 1: pisahkan file yang sudah di-cache (fast path) dari yang baru
            new_photos = []  # list of (global_idx, photo_path)
            for idx, photo_path in enumerate(photo_list):
                file_name_rel = str(photo_path.relative_to(fm.input_dir))
                if file_name_rel in processed_files:
                    processed_count += 1
                    self.on_progress(idx + 1, total_photos, f"[Cached] {photo_path.name}")
                else:
                    new_photos.append((idx, photo_path))

            # 3b. Phase 2: proses foto baru dalam chunk (detect → batch embed → cache)
            for chunk_start in range(0, len(new_photos), CHUNK_SIZE):
                if self.stopped():
                    logger.info("PipelineWorker dihentikan oleh pengguna secara paksa.")
                    return

                chunk = new_photos[chunk_start: chunk_start + CHUNK_SIZE]

                # Step A: Deteksi + alignment semua foto dalam chunk
                detected_chunk = []  # list of (file_name_rel, faces_list)
                for global_idx, photo_path in chunk:
                    if self.stopped():
                        return
                    file_name_rel = str(photo_path.relative_to(fm.input_dir))
                    try:
                        faces = detector.detect_and_crop(str(photo_path))
                    except Exception as e:
                        logger.warning(f"Gagal deteksi {photo_path.name}: {str(e)}. Dilewati.")
                        faces = []
                    detected_chunk.append((file_name_rel, faces))
                    self.on_progress(global_idx + 1, total_photos, photo_path.name)

                # Step B: Kumpulkan semua crop wajah untuk batch embedding
                all_crops = []
                metadata = []  # (file_name_rel, bbox)
                for file_name_rel, faces in detected_chunk:
                    for face_data in faces:
                        all_crops.append(face_data["cropped_face"])
                        metadata.append((file_name_rel, face_data["bbox"]))

                # Step C: Satu batch embedding call → tulis cache semua sekaligus
                if all_crops:
                    try:
                        embeddings = embedder.extract_embeddings_batch(all_crops)
                        for emb, (fname, bbox) in zip(embeddings, metadata):
                            cache_handler.simpan_cache(fname, emb, bbox)
                        new_faces_count += len(all_crops)
                        logger.info(
                            f"Batch chunk {chunk_start//CHUNK_SIZE + 1}: "
                            f"{len(all_crops)} wajah dari {len(chunk)} foto diproses."
                        )
                    except Exception as e:
                        logger.error(f"Batch embedding gagal pada chunk ini: {str(e)}")

                processed_count += len(chunk)

            # Jalankan callback selesai
            if not self.stopped():
                if drive is not None:
                    try:
                        self.on_progress(total_photos, total_photos, "Menyinkronkan cache baru ke Google Drive...")
                        _t0 = time.time()
                        drive.upload_new_cache_files(local_root)
                        drive_upload_time = time.time() - _t0
                    except Exception as e:
                        logger.error(f"Gagal menyinkronkan cache baru ke Google Drive: {str(e)}")

                timings = {
                    "detection":  detector.total_detection_time,
                    "alignment":  detector.total_alignment_time,
                    "embedding":  embedder.total_embedding_time,
                    "new_images": len(new_photos),
                    "faces_detected": len(existing_cache) + new_faces_count,
                    "new_faces_detected": new_faces_count,
                    "drive_sync_time": drive_download_time + drive_upload_time,
                }
                self.on_finished(processed_count, timings)
                
        except Exception as e:
            logger.error(f"Kegagalan fatal pada background PipelineWorker: {str(e)}")
            self.on_error(str(e))


class SearchWorker(threading.Thread):
    def __init__(self, folder_path: str, query_img_path: str, threshold: float, on_progress, on_finished, on_error, detector=None, embedder=None, drive_link: str = None):
        """
        Background worker untuk mencocokkan foto selfie kueri dengan seluruh data cache RAM.

        Args:
            folder_path (str): Jalur direktori utama tempat kumpulan foto disimpan (mode lokal).
            query_img_path (str): Jalur berkas kueri foto selfie sebagai acuan.
            threshold (float): Ambang batas jarak Cosine Distance.
            on_progress (callable): Callback untuk memperbarui log status.
            on_finished (callable): Callback saat pencarian selesai (matches, output_folder).
            on_error (callable): Callback jika terjadi galat.
            drive_link (str, optional): Tautan/ID folder Google Drive. Jika diisi, folder_path
                diabaikan dan pencarian berjalan terhadap folder Drive tersebut (lihat drive_io.DriveManager).
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
        self.drive_link = drive_link
        self.drive_sync_time = 0.0  # total waktu unduh+unggah Drive pada run ini (diisi oleh run())
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

            # 0. Jika sumber input adalah Google Drive, sinkronkan dahulu ke folder staging lokal
            drive = None
            effective_folder = self.folder_path
            if self.drive_link:
                from drive_io.drive_manager import DriveManager
                self.on_progress("Menghubungkan ke Google Drive...")
                drive = DriveManager(self.drive_link)
                local_root = Path.home() / ".personal_image_search" / "drive_cache" / drive.root_folder_id
                self.on_progress("Menyinkronkan foto & cache dari Google Drive...")
                _t0 = time.time()
                drive.download_folder_to_local(local_root)
                self.drive_sync_time += time.time() - _t0
                effective_folder = str(local_root)

            # Inisialisasi
            fm = FileManager(effective_folder)
            detector = self.detector if self.detector is not None else FaceDetector("scrfd")
            embedder = self.embedder if self.embedder is not None else ArcFaceEmbedder()
            matcher = FaceMatcher(threshold=self.threshold)
            cache_handler = CacheHandler(effective_folder)
            
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

            # Catatan: pada mode Drive, field "file_path" di cache bisa berasal dari staging
            # lokal mesin lain (cache disinkronkan turun dari Drive), sehingga tidak bisa dipercaya
            # begitu saja — path direkonstruksi dari "file_name" relatif terhadap folder staging saat ini.
            if drive is not None:
                match_paths = [str(Path(effective_folder) / item["file_name"]) for item in matches]
            else:
                match_paths = [item["file_path"] for item in matches]
            output_folder = fm.salin_hasil_cocok(match_paths)

            if drive is not None:
                try:
                    self.on_progress("Menyalin hasil pencocokan ke Google Drive...")
                    _t0 = time.time()
                    drive.copy_matches_to_drive_results([item["file_name"] for item in matches])
                    self.drive_sync_time += time.time() - _t0
                except Exception as e:
                    logger.error(f"Gagal menyalin hasil pencocokan ke Google Drive: {str(e)}")

            if not self.stopped():
                self.on_progress("Pencarian selesai dengan sukses!")
                self.on_finished(matches, output_folder)
                
        except Exception as e:
            logger.error(f"Kegagalan fatal pada background SearchWorker: {str(e)}")
            self.on_error(str(e))
