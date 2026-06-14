import os
import shutil
import logging
import numpy as np
import cv2
import onnxruntime as ort

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def configure_tensorflow():
    """
    Mengonfigurasi TensorFlow agar tidak mengalokasikan seluruh memori GPU secara serakah
    (greedy memory allocation). Mengaktifkan opsi allow_growth = True pada GPU jika tersedia.
    """
    try:
        import tensorflow as tf
        # Mengatur level log TensorFlow untuk mengurangi noise di konsol
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"TensorFlow GPU memory growth diaktifkan untuk {len(gpus)} GPU.")
        else:
            logger.info("TensorFlow berjalan pada CPU (tidak ada GPU yang terdeteksi).")
    except Exception as e:
        logger.warning(f"Gagal mengonfigurasi TensorFlow GPU memory growth: {str(e)}")

# Panggil konfigurasi TensorFlow saat modul diimpor
configure_tensorflow()

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
    # Mendapatkan path absolut root proyek berdasarkan lokasi file ini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
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

class ArcFaceEmbedder:
    def __init__(self):
        """
        Inisialisasi engine ekstraksi wajah ArcFace secara offline menggunakan ONNX Runtime.
        Mencari file model 'w600k_r50.onnx' di folder models/.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.model_path = os.path.join(project_root, "models", "w600k_r50.onnx")
        
        if not os.path.exists(self.model_path):
            # Fallback jika dijalankan dari subdirektori (misal tests/ atau gui/)
            self.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "w600k_r50.onnx"))
            
        if not os.path.exists(self.model_path):
            raise RuntimeError(
                f"File model ArcFace ONNX tidak ditemukan di: {self.model_path}. "
                "Pastikan model 'w600k_r50.onnx' ada di folder 'models/' proyek."
            )
            
        try:
            logger.info(f"Memuat model ArcFace ONNX dari: {self.model_path}...")
            # Gunakan CPU Execution Provider untuk kompatibilitas universal
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info("Model ArcFace ONNX berhasil dimuat ke memori secara eager.")
        except Exception as e:
            logger.error(f"Gagal memuat model ArcFace ONNX: {str(e)}")
            raise RuntimeError(f"Gagal menginisialisasi model ArcFace: {str(e)}")

    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Mengekstrak embedding wajah menggunakan model ArcFace ONNX berdimensi 512 dan dinormalisasi L2.
        Input diasumsikan berupa gambar wajah yang sudah dipotong (cropped) dan diselaraskan (aligned)
        oleh modul detector.py.
        
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
                
            # 3. Prapemrosesan sesuai ekspektasi InsightFace ArcFace ONNX:
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
                
            # Output shape dari model InsightFace ONNX biasanya adalah (1, 512)
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
