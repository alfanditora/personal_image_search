import os
import time
import shutil
import logging
import numpy as np
import cv2
import onnxruntime as ort

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_weights() -> bool:
    """
    Memeriksa dan memindahkan/menyalin file bobot model 'arcface_weights.h5' secara lokal
    dari direktori 'models/' proyek ke folder cache home user (~/.deepface/weights/).
    Hal ini diperlukan agar aplikasi dapat berjalan 100% secara offline tanpa mengunduh ulang.
    
    Returns:
        bool: True jika file bobot siap digunakan di folder target, False sebaliknya.
    """
    home_dir = os.path.expanduser("~")
    target_dir = os.path.join(home_dir, ".deepface", "weights")
    target_path = os.path.join(target_dir, "arcface_weights.h5")
    
    # 1. Cek apakah file bobot sudah ada di folder target cache user
    if os.path.exists(target_path):
        target_size = os.path.getsize(target_path)
        # Check jika ukuran file valid (ArcFace weights biasanya sekitar 130MB+)
        if target_size > 130 * 1024 * 1024:
            logger.info(f"Bobot model ArcFace sudah tersedia di cache target: {target_path} (Ukuran: {target_size} bytes). Proses penyalinan dilewati.")
            return True
        else:
            logger.warning(f"File bobot ditemukan di target tetapi berukuran tidak valid ({target_size} bytes). Mencoba menyalin ulang...")
            
    # 2. Cek apakah file bobot ada di folder models/ proyek
    # Mendapatkan path absolut root proyek berdasarkan lokasi file ini (src/core → src → root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    source_path = os.path.join(project_root, "models", "arcface_weights.h5")
    
    if not os.path.exists(source_path):
        logger.error(f"File bobot model 'arcface_weights.h5' tidak ditemukan di folder proyek: {source_path}")
        return False
        
    # 3. Buat folder target jika belum ada
    try:
        os.makedirs(target_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Gagal membuat direktori cache target {target_dir}: {str(e)}")
        return False
        
    # 4. Salin file dari folder models ke folder target
    try:
        logger.info(f"Menyalin file bobot model dari {source_path} ke {target_path}...")
        # Menyalin file dengan shutil.copy2 untuk mempertahankan metadata
        shutil.copy2(source_path, target_path)
        logger.info("File bobot model berhasil disalin ke cache target.")
        return True
    except Exception as e:
        logger.error(f"Gagal menyalin file bobot model ke cache target: {str(e)}")
        return False

class ArcFaceONNXRunner:
    MAX_BATCH_SIZE = 32  # cap wajah per session.run() agar peak memory tetap terkendali

    def __init__(self, model_path: str = None):
        """
        Inisialisasi engine ekstraksi wajah ArcFace menggunakan ONNX Runtime.
        Mencari file model 'w600k_r50.onnx' di folder models/.
        """
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            self.model_path = os.path.join(project_root, "models", "w600k_r50.onnx")

            if not os.path.exists(self.model_path):
                # Fallback: two levels up from src/core/
                self.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "w600k_r50.onnx"))
        else:
            self.model_path = model_path
            
        if not os.path.exists(self.model_path):
            error_msg = (
                f"File model ArcFace ONNX tidak ditemukan di: {self.model_path}. "
                "Pastikan model 'w600k_r50.onnx' tersedia di folder 'models/' proyek."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            logger.info(f"Memuat model ArcFace ONNX dari: {self.model_path}...")
            
            # Optimasi SessionOptions untuk mencegah CPU starvation dan memory crash
            session_options = ort.SessionOptions()
            session_options.enable_mem_pattern = True
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            # Menggunakan setelan thread bawaan ONNX Runtime (tidak dibatasi 50%)
            logger.info("Menggunakan setelan thread default untuk ONNX Runtime.")
            
            self.session = ort.InferenceSession(
                self.model_path, 
                sess_options=session_options, 
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info("Model ArcFace ONNX berhasil dimuat.")
        except Exception as e:
            logger.error(f"Gagal memuat model ArcFace ONNX: {str(e)}")
            raise RuntimeError(f"Gagal menginisialisasi model ArcFace: {str(e)}")

        # Timing accumulator (reset per indexing run via reset_timings())
        self.total_embedding_time = 0.0

    def reset_timings(self):
        self.total_embedding_time = 0.0

    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Mengekstrak embedding wajah menggunakan model ArcFace ONNX berdimensi 512 dan dinormalisasi L2.
        Input berupa gambar wajah (112x112 BGR).
        
        Args:
            face_image (np.ndarray): NumPy array berisi citra wajah (112x112 BGR).
            
        Returns:
            np.ndarray: Vektor embedding 512-dimensi yang sudah dinormalisasi L2.
        """
        if face_image is None or face_image.size == 0:
            raise ValueError("Input citra wajah tidak boleh kosong (None atau berukuran 0).")
            
        try:
            # 1. Konversi format BGR (OpenCV) ke RGB
            if len(face_image.shape) == 2:
                img_rgb = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            elif face_image.shape[2] == 4:
                img_rgb = cv2.cvtColor(face_image, cv2.COLOR_RGBA2RGB)
            else:
                img_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                
            # 2. Resize ke 112x112 jika dimensinya belum pas
            if img_rgb.shape[0] != 112 or img_rgb.shape[1] != 112:
                img_rgb = cv2.resize(img_rgb, (112, 112), interpolation=cv2.INTER_AREA)
                
            # 3. Prapemrosesan InsightFace ArcFace ONNX:
            # - Normalisasi: (img - 127.5) / 127.5
            # - Transpose dari HWC (112, 112, 3) menjadi CHW (3, 112, 112)
            # - Expand dimensions ke batch (1, 3, 112, 112)
            img_input = img_rgb.astype(np.float32)
            img_input = (img_input - 127.5) / 127.5
            img_input = np.transpose(img_input, (2, 0, 1))
            img_input = np.expand_dims(img_input, axis=0)
            
            # 4. Inferensi model ONNX
            embeddings = self.session.run([self.output_name], {self.input_name: img_input})
            
            if not embeddings:
                raise RuntimeError("ONNX Runtime mengembalikan output embedding kosong.")
                
            vector = np.array(embeddings[0][0], dtype=np.float32)
            
            # 5. Menerapkan L2 Normalization secara manual
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector_l2 = vector / norm
            else:
                vector_l2 = vector
                
            return vector_l2
            
        except Exception as e:
            logger.error(f"Gagal mengekstrak embedding wajah menggunakan ONNX: {str(e)}")
            raise RuntimeError(f"Gagal mengekstrak embedding: {str(e)}")

    def extract_embeddings_batch(self, face_images: list) -> np.ndarray:
        """
        Ekstrak embedding untuk beberapa wajah sekaligus dalam satu atau lebih panggilan
        session.run(). Batch besar (mis. foto grup berisi banyak wajah dalam satu chunk)
        dipecah menjadi sub-batch berukuran maksimum MAX_BATCH_SIZE agar buffer internal
        ONNX Runtime (mis. node ReorderOutput) tidak meminta alokasi memori raksasa
        sekaligus dan memicu BFCArena OOM.

        Args:
            face_images (list[np.ndarray]): Daftar citra wajah 112×112 BGR.

        Returns:
            np.ndarray: Matriks (N, 512) embedding L2-normalized.
        """
        if not face_images:
            return np.zeros((0, 512), dtype=np.float32)

        results = [
            self._run_batch_safe(face_images[start:start + self.MAX_BATCH_SIZE])
            for start in range(0, len(face_images), self.MAX_BATCH_SIZE)
        ]
        embeddings = np.concatenate(results, axis=0)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def _preprocess_batch(self, face_images: list) -> np.ndarray:
        batch = []
        for face_image in face_images:
            if len(face_image.shape) == 2:
                img = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            elif face_image.shape[2] == 4:
                img = cv2.cvtColor(face_image, cv2.COLOR_RGBA2RGB)
            else:
                img = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            if img.shape[0] != 112 or img.shape[1] != 112:
                img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)

            img = img.astype(np.float32)
            img = (img - 127.5) / 127.5
            img = np.transpose(img, (2, 0, 1))  # CHW
            batch.append(img)

        return np.stack(batch, axis=0)  # (N, 3, 112, 112)

    def _run_batch_safe(self, face_images: list) -> np.ndarray:
        """
        Menjalankan satu sub-batch, dan jika ONNX Runtime gagal mengalokasikan memori
        (mis. BFCArena AllocateRawInternal), coba lagi dengan sub-batch dibagi dua secara
        rekursif hingga ukuran 1 sebelum akhirnya menyerah (meniru pola retry resolusi
        bertahap pada FaceDetector._detect_faces_safe()).
        """
        batch_tensor = self._preprocess_batch(face_images)
        try:
            _t = time.perf_counter()
            raw = self.session.run([self.output_name], {self.input_name: batch_tensor})
            self.total_embedding_time += time.perf_counter() - _t
            return np.array(raw[0], dtype=np.float32)  # (N, 512)
        except Exception as e:
            if len(face_images) > 1:
                logger.warning(
                    f"Batch embedding ukuran {len(face_images)} gagal ({str(e)}), "
                    f"mencoba ulang dengan sub-batch lebih kecil."
                )
                mid = len(face_images) // 2
                first = self._run_batch_safe(face_images[:mid])
                second = self._run_batch_safe(face_images[mid:])
                return np.concatenate([first, second], axis=0)
            logger.error(f"Gagal batch embedding: {str(e)}")
            raise RuntimeError(f"Batch embedding gagal: {str(e)}")


# Alias untuk backward compatibility
ArcFaceEmbedder = ArcFaceONNXRunner
