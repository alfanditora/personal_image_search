import logging
import numpy as np

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceMatcher:
    def __init__(self, threshold: float = 0.40):
        """
        Menginisialisasi FaceMatcher dengan parameter ambang batas kemiripan.
        
        Args:
            threshold (float): Batas toleransi jarak Cosine (Cosine Distance).
                               Untuk ArcFace, default yang direkomendasikan adalah 0.40.
                               Wajah dianggap cocok jika jarak Cosine <= threshold.
        """
        self.threshold = threshold
        logger.info(f"FaceMatcher diinisialisasi dengan Cosine Distance threshold: {self.threshold}")

    def hitung_kesamaan_cosine(self, vektor_kueri: np.ndarray, vektor_target: np.ndarray) -> float:
        """
        Menghitung nilai jarak Cosine (Cosine Distance) antara dua vektor embedding.
        Formula: Cosine Distance = 1.0 - Cosine Similarity
        
        Args:
            vektor_kueri (np.ndarray): Vektor query 512-dimensi.
            vektor_target (np.ndarray): Vektor target 512-dimensi.
            
        Returns:
            float: Nilai jarak Cosine dalam rentang [0.0, 2.0].
                   Nilai mendekati 0.0 menunjukkan kemiripan identik.
                   Nilai mendekati 1.0 menunjukkan orthogonal (tidak mirip).
                   Kembalikan 1.0 jika salah satu vektor bernilai nol (zero magnitude).
        """
        if vektor_kueri is None or vektor_target is None:
            return 1.0
            
        # Pengecekan dimensi dasar
        if vektor_kueri.shape != (512,) or vektor_target.shape != (512,):
            logger.warning("Dimensi vektor tidak sesuai, diharapkan (512,).")
            return 1.0
            
        try:
            norm_q = np.linalg.norm(vektor_kueri)
            norm_t = np.linalg.norm(vektor_target)
            
            # Mitigasi pembagian dengan nol jika magnitude vektor 0
            if norm_q == 0.0 or norm_t == 0.0:
                return 1.0
                
            similarity = np.dot(vektor_kueri, vektor_target) / (norm_q * norm_t)
            
            # Rentang numerik float safety (jaga agar nilai similarity tetap di [-1.0, 1.0])
            similarity = np.clip(similarity, -1.0, 1.0)
            
            cosine_distance = 1.0 - similarity
            return float(cosine_distance)
            
        except Exception as e:
            logger.error(f"Terjadi galat saat menghitung jarak Cosine: {str(e)}")
            return 1.0

    def cari_foto_cocok(self, vektor_selfie: np.ndarray, daftar_cache: list[dict]) -> list[dict]:
        """
        Mencari dan menyaring foto wajah yang cocok berdasarkan query vektor selfie 
        terhadap seluruh data cache wajah di dalam RAM secara ter-vektorisasi (batch).
        
        Memenuhi parameter stabilitas NF3 (tidak crash pada data rusak) dan performa NF1 (< 1 detik).
        
        Args:
            vektor_selfie (np.ndarray): Vektor embedding wajah kueri (selfie) berdimensi 512.
            daftar_cache (list[dict]): List metadata cache wajah yang dimuat ke memori RAM.
            
        Returns:
            list[dict]: List metadata wajah yang cocok (lolos threshold), 
                        diurutkan dari kemiripan tertinggi (Cosine Distance terkecil).
                        Setiap item menyertakan nilai 'cosine_distance' dan 'similarity'.
        """
        if vektor_selfie is None or not daftar_cache:
            return []
            
        # 1. Validasi dimensi vektor selfie kueri
        if vektor_selfie.shape != (512,):
            logger.error(f"Dimensi vektor selfie kueri tidak sesuai: {vektor_selfie.shape}, diharapkan (512,).")
            return []
            
        norm_selfie = np.linalg.norm(vektor_selfie)
        if norm_selfie == 0.0:
            logger.error("Vektor selfie memiliki magnitudo nol.")
            return []
            
        # Normalisasi vektor selfie kueri
        selfie_norm = vektor_selfie / norm_selfie
        
        # 2. Saring data cache yang valid secara struktur untuk komputasi matriks ter-vektorisasi
        valid_items = []
        target_vectors = []
        
        for idx, item in enumerate(daftar_cache):
            try:
                vec = item.get("embedding")
                # Validasi tipe data dan dimensi array target
                if vec is not None and isinstance(vec, np.ndarray) and vec.shape == (512,):
                    valid_items.append(item)
                    target_vectors.append(vec)
                else:
                    # Lewati data rusak secara mandiri tanpa crash (NF3)
                    logger.warning(f"Data cache indeks ke-{idx} dilewati karena format vektor embedding salah/None.")
            except Exception as e:
                logger.error(f"Galat saat memvalidasi item cache ke-{idx}: {str(e)}")
                continue
                
        if not target_vectors:
            logger.info("Tidak ada target vektor valid untuk dicocokkan.")
            return []
            
        try:
            # 3. Lakukan komputasi matriks ter-vektorisasi menggunakan NumPy untuk efisiensi tinggi (NF1)
            targets_matrix = np.stack(target_vectors)  # Dimensi (N, 512)
            
            # Hitung norm masing-masing vektor target
            norms_targets = np.linalg.norm(targets_matrix, axis=1, keepdims=True)
            
            # Tangani jika ada target dengan magnitude 0 (hindari pembagian nol)
            norms_targets[norms_targets == 0.0] = 1.0
            
            # Normalisasi matriks target
            targets_matrix_norm = targets_matrix / norms_targets
            
            # Hitung Cosine Similarity serentak menggunakan dot product (N, 512) x (512,) -> (N,)
            similarities = np.dot(targets_matrix_norm, selfie_norm)
            
            # Jaga kestabilan batas numerik
            similarities = np.clip(similarities, -1.0, 1.0)
            
            # Hitung Cosine Distance: 1.0 - Cosine Similarity
            distances = 1.0 - similarities
            
            # 4. Saring hasil yang memenuhi ambang batas (jarak <= threshold)
            matches = []
            for i, dist in enumerate(distances):
                if dist <= self.threshold:
                    matched_item = dict(valid_items[i])
                    # Tambahkan parameter jarak untuk digunakan oleh GUI
                    matched_item["cosine_distance"] = float(dist)
                    matched_item["similarity"] = float(similarities[i])
                    matches.append(matched_item)
            
            # 5. Urutkan hasil dari jarak terkecil (kemiripan tertinggi) ke jarak terbesar
            matches = sorted(matches, key=lambda x: x["cosine_distance"])
            
            logger.info(f"Proses pencocokan selesai. Menemukan {len(matches)} wajah yang cocok dari {len(daftar_cache)} data.")
            return matches
            
        except Exception as e:
            logger.error(f"Gagal melakukan proses pencarian foto cocok secara ter-vektorisasi: {str(e)}")
            return []
