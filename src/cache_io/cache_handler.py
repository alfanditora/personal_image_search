import os
import json
import hashlib
import logging
import numpy as np

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CacheHandler:
    def __init__(self, folder_input_path: str):
        """
        Menginisialisasi CacheHandler untuk folder input tertentu.
        Membuat subdirektori tersembunyi '.face_cache' untuk penyimpanan data Zero-DB.
        
        Args:
            folder_input_path (str): Jalur direktori utama tempat kumpulan foto disimpan.
        """
        if not folder_input_path:
            raise ValueError("Jalur folder input tidak boleh kosong.")
            
        self.folder_input_path = os.path.abspath(folder_input_path)
        self.cache_dir = os.path.join(self.folder_input_path, ".face_cache")
        
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Direktori cache Zero-DB siap di: {self.cache_dir}")
        except Exception as e:
            logger.error(f"Gagal menginisialisasi direktori cache {self.cache_dir}: {str(e)}")
            raise RuntimeError(f"Gagal membuat direktori cache: {str(e)}")

    def simpan_cache(self, file_name: str, embedding: np.ndarray, bbox: tuple) -> bool:
        """
        Menyimpan matriks vektor embedding wajah (512-dimensi) beserta metadatanya
        ke berkas cache lokal (.json) secara aman dengan penamaan berbasis MD5 hash unik.
        
        Args:
            file_name (str): Nama file gambar asli relatif terhadap folder input (misal: "foto1.png").
            embedding (np.ndarray): Vektor fitur wajah NumPy array berdimensi 512.
            bbox (tuple): Koordinat bounding box wajah (x, y, w, h).
            
        Returns:
            bool: True jika penyimpanan berhasil, False jika terjadi kegagalan (hak akses/disk penuh).
        """
        if embedding is None or len(embedding) == 0:
            logger.warning("Vektor embedding kosong, proses simpan cache dibatalkan.")
            return False
            
        try:
            # 1. Konversi embedding NumPy array ke list agar bisa diserialisasikan ke JSON
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
            
            # 2. Buat ID unik berbasis MD5 dari gabungan nama berkas asli dan bbox koordinat
            bbox_str = f"_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
            unique_id = hashlib.md5((file_name + bbox_str).encode('utf-8')).hexdigest()
            
            # 3. Bentuk skema metadata data wajah
            metadata = {
                "id": unique_id,
                "file_name": file_name,
                "file_path": os.path.join(self.folder_input_path, file_name),
                "bbox": list(bbox),
                "embedding": embedding_list
            }
            
            # 4. Tentukan lokasi berkas simpan cache
            cache_file_path = os.path.join(self.cache_dir, f"{unique_id}.json")
            
            # 5. Lakukan penulisan data secara aman
            with open(cache_file_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
                
            return True
            
        except PermissionError as e:
            # Penanganan hak akses terbatas (read-only)
            logger.error(f"Gagal menulis berkas cache karena keterbatasan hak akses (PermissionError): {str(e)}")
            return False
        except OSError as e:
            # Penanganan low disk space / OSError
            logger.error(f"Gagal menulis berkas cache karena ruang penyimpanan penuh atau galat sistem (OSError): {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Terjadi galat tak terduga saat menyimpan cache wajah untuk {file_name}: {str(e)}")
            return False

    def muat_seluruh_cache(self) -> list[dict]:
        """
        Memuat seluruh data cache wajah (.json) yang tersimpan di dalam folder tersembunyi
        kembali ke dalam memori RAM secara sekuensial (batch processing).
        
        Memiliki mekanisme skip berkas rusak (corrupt) agar pemuatan berkas valid lainnya terus berjalan (NF3).
        
        Returns:
            list[dict]: List of dictionaries berisi metadata wajah di mana vektor embedding 
                        telah dipulihkan kembali ke bentuk NumPy array (np.ndarray, np.float32).
        """
        results = []
        
        if not os.path.exists(self.cache_dir):
            logger.warning(f"Direktori cache tidak ditemukan: {self.cache_dir}")
            return []
            
        # Pindai seluruh berkas di folder cache
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith(".json")]
            logger.info(f"Menemukan {len(cache_files)} berkas cache di {self.cache_dir}. Memulai pemuatan...")
        except Exception as e:
            logger.error(f"Gagal membaca daftar berkas di folder cache: {str(e)}")
            return []
            
        for file_name in cache_files:
            file_path = os.path.join(self.cache_dir, file_name)
            try:
                # 1. Baca dan urai berkas JSON
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                # 2. Validasi struktur metadata minimal
                required_keys = ["id", "file_name", "file_path", "bbox", "embedding"]
                if not all(k in data for k in required_keys):
                    logger.warning(f"Berkas cache {file_name} dilewati karena struktur metadata tidak lengkap.")
                    continue
                    
                # 3. Pulihkan list embedding kembali ke format NumPy array berpresisi float32
                data["embedding"] = np.array(data["embedding"], dtype=np.float32)
                
                results.append(data)
                
            except json.JSONDecodeError as e:
                # Tangkap jika JSON corrupt / rusak
                logger.error(f"Berkas cache {file_name} rusak (JSONDecodeError): {str(e)}. Melewati berkas ini.")
                continue
            except Exception as e:
                # Tangkap galat lainnya agar pemuatan batch tidak crash secara keseluruhan (NF3)
                logger.error(f"Gagal memuat berkas cache {file_name}: {str(e)}. Melewati berkas ini.")
                continue
                
        logger.info(f"Berhasil memuat {len(results)} data cache wajah yang valid ke dalam RAM.")
        return results
