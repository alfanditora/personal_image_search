import os
import cv2
import logging
import numpy as np
from mtcnn import MTCNN

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, detector_type: str = "mtcnn"):
        """
        Menginisialisasi detektor wajah (MTCNN atau SCRFD).
        
        Args:
            detector_type (str): Tipe detektor, bisa "mtcnn" atau "scrfd".
        """
        self.detector_type = detector_type.lower()
        if self.detector_type == "mtcnn":
            logger.info("Menginisialisasi detektor wajah MTCNN...")
            try:
                self.detector = MTCNN()
                logger.info("Detektor wajah MTCNN berhasil diinisialisasi.")
            except Exception as e:
                logger.error(f"Gagal menginisialisasi MTCNN: {str(e)}")
                raise RuntimeError(f"Gagal menginisialisasi MTCNN: {str(e)}")
        elif self.detector_type == "scrfd":
            logger.info("Menginisialisasi detektor wajah SCRFD...")
            try:
                from scrfd import SCRFD
                # Gunakan model det_2.5g.onnx yang tersimpan di folder models
                model_path = os.path.join("models", "det_2.5g.onnx")
                if not os.path.exists(model_path):
                    # Fallback jika dijalankan dari subdirektori (misal tests/ atau algorithm_exploration/)
                    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "det_2.5g.onnx"))
                
                self.detector = SCRFD.from_path(model_path)
                logger.info(f"Detektor wajah SCRFD berhasil diinisialisasi dari {model_path}.")
            except Exception as e:
                logger.error(f"Gagal menginisialisasi SCRFD: {str(e)}")
                raise RuntimeError(f"Gagal menginisialisasi SCRFD: {str(e)}")
        else:
            raise ValueError(f"Tipe detektor tidak dikenal: {detector_type}. Harus 'mtcnn' atau 'scrfd'.")

    def _detect_faces_safe(self, img: np.ndarray, initial_max_dim: int = 960) -> tuple[list, float]:
        """
        Mendeteksi wajah secara aman dengan fallback otomatis ke resolusi lebih rendah jika terjadi OOM.
        Mendukung detektor MTCNN dan SCRFD secara seragam.
        
        Args:
            img (np.ndarray): Matriks citra input (BGR).
            initial_max_dim (int): Dimensi maksimal awal untuk downscaling.
            
        Returns:
            tuple[list, float]: (daftar wajah terdeteksi dengan format MTCNN standar, skala gambar relatif)
        """
        import gc
        orig_h, orig_w = img.shape[:2]
        
        # Cobalah beberapa tingkat resolusi secara bertahap jika gagal alokasi memori
        # Batasi resolusi awal maksimal 960px untuk menghemat CPU & memori tanpa mengorbankan akurasi
        resolutions = [initial_max_dim, 720, 512, 360]
        
        for idx, target_max_dim in enumerate(resolutions):
            try:
                scale = 1.0
                if max(orig_h, orig_w) > target_max_dim:
                    scale = target_max_dim / max(orig_h, orig_w)
                    target_w = int(orig_w * scale)
                    target_h = int(orig_h * scale)
                    img_detect = cv2.resize(img, (target_w, target_h))
                else:
                    img_detect = img
                    scale = 1.0
                
                if self.detector_type == "mtcnn":
                    # Gunakan min_face_size = 20
                    faces = self.detector.detect_faces(img_detect, min_face_size=20)
                    return faces, scale
                elif self.detector_type == "scrfd":
                    from PIL import Image
                    from scrfd import Threshold
                    # Convert cv2 image (BGR) to PIL Image (RGB)
                    img_rgb = cv2.cvtColor(img_detect, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    
                    # Jalankan deteksi SCRFD (threshold confidence = 0.5)
                    threshold = Threshold(probability=0.5)
                    faces_scrfd = self.detector.detect(pil_img, threshold=threshold)
                    
                    # Normalisasi output SCRFD ke format list of dict standar MTCNN
                    faces = []
                    for face in faces_scrfd:
                        x = int(face.bbox.upper_left.x)
                        y = int(face.bbox.upper_left.y)
                        w = int(face.bbox.lower_right.x - x)
                        h = int(face.bbox.lower_right.y - y)
                        
                        faces.append({
                            'box': [x, y, w, h],
                            'keypoints': {
                                'left_eye': [face.keypoints.left_eye.x, face.keypoints.left_eye.y],
                                'right_eye': [face.keypoints.right_eye.x, face.keypoints.right_eye.y],
                                'nose': [face.keypoints.nose.x, face.keypoints.nose.y],
                                'mouth_left': [face.keypoints.left_mouth.x, face.keypoints.left_mouth.y],
                                'mouth_right': [face.keypoints.right_mouth.x, face.keypoints.right_mouth.y]
                            }
                        })
                    return faces, scale
                    
            except Exception as e:
                error_msg = str(e).lower()
                # Tangani galat kehabisan memori (OOM / allocation error)
                if "allocate" in error_msg or "memory" in error_msg or "oom" in error_msg:
                    logger.warning(
                        f"Percobaan deteksi ke-{idx+1} gagal (OOM) pada resolusi maks {target_max_dim}px: {str(e)}. "
                        "Memaksa pembersihan memori (garbage collection) dan mencoba resolusi lebih rendah..."
                    )
                    gc.collect()
                    continue
                else:
                    raise e
                    
        logger.error("Semua percobaan resolusi untuk deteksi wajah gagal karena masalah memori.")
        return [], 1.0

    def align_face(self, face_img: np.ndarray) -> np.ndarray:
        """
                Penyelarasan wajah (face alignment) berbasis Similarity Transform 2D (5-titik landmark).
        Memetakan mata, hidung, dan ujung mulut ke koordinat template standar 112x112.
        
        Args:
            face_img (np.ndarray): Matriks citra potongan wajah (BGR).
            
        Returns:
            np.ndarray: Matriks citra wajah yang sudah diselaraskan ke resolusi 112x112.
        """
        if face_img is None or face_img.size == 0:
            return face_img
            
        # Jika gambar terlalu kecil (di bawah 200px), asumsikan ini adalah potongan wajah (crop)
        # yang sudah dipotong rapat, sehingga kita lewati deteksi ulang untuk efisiensi tinggi.
        if max(face_img.shape[:2]) < 200:
            return face_img
            
        try:
            # Menggunakan deteksi aman dengan max_dim awal 1024 untuk krop wajah besar
            faces, scale = self._detect_faces_safe(face_img, initial_max_dim=1024)
            if faces:
                keypoints = faces[0]['keypoints']
                required_keys = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']
                if all(k in keypoints for k in required_keys):
                    # Petakan koordinat deteksi kembali ke skala asli face_img
                    src_pts = np.array([
                        [keypoints['left_eye'][0] / scale, keypoints['left_eye'][1] / scale],
                        [keypoints['right_eye'][0] / scale, keypoints['right_eye'][1] / scale],
                        [keypoints['nose'][0] / scale, keypoints['nose'][1] / scale],
                        [keypoints['mouth_left'][0] / scale, keypoints['mouth_left'][1] / scale],
                        [keypoints['mouth_right'][0] / scale, keypoints['mouth_right'][1] / scale]
                    ], dtype=np.float32)
                    
                    # Target template standar 112x112 (InsightFace / MTCNN)
                    dst_pts = np.array([
                        [38.2946, 51.6963],
                        [73.5318, 51.5014],
                        [56.0252, 71.7366],
                        [41.5493, 92.3655],
                        [70.7299, 92.2041]
                    ], dtype=np.float32)
                    
                    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                    if M is not None:
                        aligned_face = cv2.warpAffine(face_img, M, (112, 112), flags=cv2.INTER_CUBIC)
                        return aligned_face
            return face_img
            
        except Exception as e:
            logger.warning(f"Galat selama proses penyelarasan wajah (alignment): {str(e)}. Menggunakan fallback.")
            return face_img

    def detect_and_crop(self, img_path: str) -> list[dict]:
        """
        Mendeteksi wajah pada gambar tunggal, memotongnya, menyelaraskan orientasinya,
        dan mengubah ukurannya menjadi 112x112 piksel secara konsisten menggunakan MTCNN.
        
        Melakukan downscaling gambar untuk proses deteksi guna mencegah masalah kehabisan memori (OOM)
        dan kelambatan performa pada resolusi ultra-tinggi (misal 8MP+), lalu memotong wajah 
        berkualitas tinggi dari citra asli beresolusi penuh.
        
        Args:
            img_path (str): Jalur lengkap file gambar.
            
        Returns:
            list[dict]: List berisi dictionary dengan kunci:
                - "cropped_face": NumPy array gambar wajah berukuran 112x112 (BGR).
                - "bbox": Tuple koordinat kotak pembatas (x, y, w, h).
        """
        results = []
        
        # 1. Validasi keberadaan file secara fisik
        if not img_path or not os.path.exists(img_path):
            logger.warning(f"Berkas gambar tidak ditemukan atau path kosong: {img_path}")
            return []
            
        try:
            # 2. Muat gambar asli beresolusi tinggi menggunakan OpenCV
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Gagal memuat gambar (Format corrupt atau tidak didukung): {img_path}")
                return []
                
            orig_h, orig_w = img.shape[:2]
            
            # 3. Deteksi wajah secara aman & adaptif terhadap memori (menggunakan helper _detect_faces_safe)
            faces, scale = self._detect_faces_safe(img, initial_max_dim=960)
            
            # 4. Iterasi dan prapemrosesan masing-masing wajah terdeteksi
            for face in faces:
                x, y, w, h = face['box']
                
                # Petakan kembali koordinat bounding box ke dimensi citra asli beresolusi tinggi
                x_orig = int(x / scale)
                y_orig = int(y / scale)
                w_orig = int(w / scale)
                h_orig = int(h / scale)
                
                # Dapatkan koordinat 5 keypoints dan petakan kembali ke citra asli beresolusi tinggi
                keypoints = face['keypoints']
                required_keys = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']
                
                if all(k in keypoints for k in required_keys):
                    src_pts = np.array([
                        [keypoints['left_eye'][0] / scale, keypoints['left_eye'][1] / scale],
                        [keypoints['right_eye'][0] / scale, keypoints['right_eye'][1] / scale],
                        [keypoints['nose'][0] / scale, keypoints['nose'][1] / scale],
                        [keypoints['mouth_left'][0] / scale, keypoints['mouth_left'][1] / scale],
                        [keypoints['mouth_right'][0] / scale, keypoints['mouth_right'][1] / scale]
                    ], dtype=np.float32)
                    
                    # Target template standar 112x112 (InsightFace / MTCNN)
                    dst_pts = np.array([
                        [38.2946, 51.6963],  # left eye
                        [73.5318, 51.5014],  # right eye
                        [56.0252, 71.7366],  # nose
                        [41.5493, 92.3655],  # left mouth
                        [70.7299, 92.2041]   # right mouth
                    ], dtype=np.float32)
                    
                    # Hitung similarity transform matrix dari citra asli ke target 112x112
                    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                else:
                    M = None
                
                if M is not None:
                    # Lakukan warp langsung dari citra asli berkualitas tinggi ke 112x112
                    aligned_face = cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_CUBIC)
                else:
                    # Fallback ke cropping sederhana jika transformasi gagal
                    x1, y1 = max(0, x_orig), max(0, y_orig)
                    x2, y2 = min(orig_w, x_orig + w_orig), min(orig_h, y_orig + h_orig)
                    face_roi = img[y1:y2, x1:x2]
                    aligned_face = cv2.resize(face_roi, (112, 112), interpolation=cv2.INTER_AREA)
                
                # Tambahkan ke list hasil
                results.append({
                    "cropped_face": aligned_face,
                    "bbox": (int(x_orig), int(y_orig), int(w_orig), int(h_orig))
                })
                
            return results
            
        except Exception as e:
            # Mitigasi crash: catat log kesalahan dan kembalikan list kosong
            logger.error(f"Terjadi galat saat memproses gambar {img_path}: {str(e)}")
            return []
