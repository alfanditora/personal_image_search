import os
import re
import logging
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import httplib2
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_httplib2 import AuthorizedHttp
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Folder yang sama seperti FileManager._EXCLUDED_DIRS / CacheHandler,
# tetapi versi Drive: keduanya subfolder khusus di dalam folder Drive input.
CACHE_FOLDER_NAME = ".face_cache"
RESULTS_FOLDER_NAME = "Hasil_Pencarian_Selfie"

_IMAGE_MIME_TYPES = {"image/jpeg", "image/png"}
_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CREDENTIALS_PATH = _PROJECT_ROOT / "models" / "credentials.json"
_TOKEN_PATH = _PROJECT_ROOT / "models" / "token.json"

SCOPES = ["https://www.googleapis.com/auth/drive"]

# Both cache-file uploads and result-photo copies are many small, independent, blocking
# Drive API calls (one network round-trip each) — sequentially that's the real bottleneck,
# not the API quota. A small worker pool overlaps those round-trips; num_retries absorbs
# transient rate-limit/backoff errors (403/429/5xx) via googleapiclient's exponential backoff.
DRIVE_POOL_WORKERS = 8
# Submitted in bounded batches rather than all at once, so a huge cache folder or a huge
# match count doesn't queue thousands of Future/work-item objects in RAM at the same time.
DRIVE_BATCH_SIZE = 200
_API_NUM_RETRIES = 5

# Photos, unlike the tiny cache JSON, can be several MB each — downloads stream straight to
# disk in chunks of this size (see _download_file) instead of buffering a whole file in RAM,
# so peak memory across DRIVE_POOL_WORKERS concurrent downloads stays bounded (~a few tens of MB)
# regardless of photo size.
DRIVE_DOWNLOAD_CHUNK_SIZE = 5 * 1024 * 1024

_FOLDER_LINK_RE = re.compile(r"/folders/([a-zA-Z0-9_-]+)")


def _escape_query_value(value: str) -> str:
    """Meng-escape tanda kutip tunggal agar aman dipakai di dalam query Drive API."""
    return value.replace("'", "\\'")


class DriveManager:
    def __init__(self, folder_link_or_id: str):
        """
        Menginisialisasi DriveManager untuk sebuah folder Google Drive tertentu,
        lalu melakukan autentikasi OAuth 2.0 (Desktop app) ke akun Google pengguna.

        Args:
            folder_link_or_id (str): Tautan folder Google Drive
                (mis. https://drive.google.com/drive/folders/<ID>) atau ID folder mentah.
        """
        if not folder_link_or_id:
            raise ValueError("Tautan/ID folder Google Drive tidak boleh kosong.")

        self.root_folder_id = self._extract_folder_id(folder_link_or_id)
        self._credentials = None  # diisi oleh _authenticate(), dipakai ulang untuk http per-thread
        self.service = self._authenticate()
        self._thread_local = threading.local()

        # Peta relative_path -> file_id, diisi oleh list_images().
        self._image_map: dict[str, str] = {}
        self._cache_folder_id: str | None = None
        self._results_folder_id: str | None = None

        logger.info(f"DriveManager siap untuk folder Drive: {self.root_folder_id}")

    @staticmethod
    def _extract_folder_id(folder_link_or_id: str) -> str:
        match = _FOLDER_LINK_RE.search(folder_link_or_id)
        if match:
            return match.group(1)
        # Anggap sudah berupa ID mentah (bukan URL) jika tidak cocok pola link folder.
        return folder_link_or_id.strip()

    def _authenticate(self):
        """
        Melakukan autentikasi OAuth 2.0 (Installed App flow) menggunakan
        models/credentials.json, dengan token hasil login di-cache ke models/token.json
        agar pengguna tidak perlu login ulang di setiap sesi.
        """
        if not _CREDENTIALS_PATH.exists():
            raise FileNotFoundError(
                f"credentials.json tidak ditemukan di {_CREDENTIALS_PATH}. "
                "Unduh OAuth Client (Desktop app) dari Google Cloud Console dan letakkan di sana."
            )

        creds = None
        if _TOKEN_PATH.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(_TOKEN_PATH), SCOPES)
            except Exception as e:
                logger.warning(f"Gagal memuat token.json yang tersimpan, akan login ulang: {str(e)}")
                creds = None

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                logger.info("Membuka browser untuk login Google Drive (OAuth)...")
                flow = InstalledAppFlow.from_client_secrets_file(str(_CREDENTIALS_PATH), SCOPES)
                creds = flow.run_local_server(port=0)

            _TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
            logger.info(f"Token autentikasi Google Drive disimpan di: {_TOKEN_PATH}")

        self._credentials = creds
        return build("drive", "v3", credentials=creds)

    def _thread_http(self):
        """
        Objek http terotorisasi khusus thread yang sedang berjalan.

        self.service tidak thread-safe untuk dipakai bersamaan dari banyak thread
        (googleapiclient berbagi satu koneksi http internal per service). Agar upload
        cache bisa dijalankan konkuren, tiap worker thread memakai httplib2.Http()
        miliknya sendiri, tetap terotorisasi dengan credentials yang sama, dan
        dioper ke execute(http=...) alih-alih memakai http bawaan service.
        """
        if not hasattr(self._thread_local, "http"):
            self._thread_local.http = AuthorizedHttp(self._credentials, http=httplib2.Http())
        return self._thread_local.http

    def _run_in_pool(self, items: list, work_fn, label_fn, thread_name_prefix: str) -> tuple[int, list]:
        """
        Menjalankan work_fn(item) untuk tiap item di 'items' memakai worker pool tetap
        (DRIVE_POOL_WORKERS), disubmit dalam batch terbatas (DRIVE_BATCH_SIZE) alih-alih
        sekaligus semua — supaya daftar kerja yang sangat besar tidak menumpuk ribuan
        objek Future/work-item di RAM secara bersamaan.

        Returns:
            tuple[int, list]: (jumlah sukses, daftar label item yang gagal via label_fn).
        """
        succeeded = 0
        failed = []
        with ThreadPoolExecutor(max_workers=DRIVE_POOL_WORKERS, thread_name_prefix=thread_name_prefix) as pool:
            for batch_start in range(0, len(items), DRIVE_BATCH_SIZE):
                batch = items[batch_start:batch_start + DRIVE_BATCH_SIZE]
                futures = {pool.submit(work_fn, item): item for item in batch}
                for future in as_completed(futures):
                    item = futures[future]
                    try:
                        future.result()
                        succeeded += 1
                    except Exception as e:
                        label = label_fn(item)
                        failed.append(label)
                        logger.error(f"Gagal memproses '{label}': {str(e)}")
        return succeeded, failed

    def _list_children(self, parent_id: str, extra_query: str = "") -> list[dict]:
        """Mengambil seluruh child (file/folder) langsung dari sebuah folder Drive, dengan paginasi."""
        query = f"'{parent_id}' in parents and trashed=false"
        if extra_query:
            query += f" and {extra_query}"

        files = []
        page_token = None
        while True:
            response = self.service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token,
                pageSize=1000,
            ).execute(num_retries=_API_NUM_RETRIES)
            files.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return files

    def _find_child_folder(self, parent_id: str, name: str) -> str | None:
        matches = self._list_children(
            parent_id,
            extra_query=f"name = '{_escape_query_value(name)}' and mimeType = '{_FOLDER_MIME_TYPE}'",
        )
        return matches[0]["id"] if matches else None

    def _get_or_create_subfolder(self, parent_id: str, name: str) -> str:
        folder_id = self._find_child_folder(parent_id, name)
        if folder_id:
            return folder_id

        metadata = {"name": name, "mimeType": _FOLDER_MIME_TYPE, "parents": [parent_id]}
        created = self.service.files().create(body=metadata, fields="id").execute(num_retries=_API_NUM_RETRIES)
        logger.info(f"Membuat folder '{name}' baru di Drive.")
        return created["id"]

    def list_images(self) -> dict[str, str]:
        """
        Menyusuri folder Drive input secara rekursif untuk menemukan seluruh gambar (jpg/png),
        sambil mengabaikan subfolder khusus '.face_cache' dan 'Hasil_Pencarian_Selfie'
        (setara dengan FileManager._EXCLUDED_DIRS pada input lokal).

        Returns:
            dict[str, str]: Peta {relative_path: file_id}.
        """
        self._image_map = {}
        stack = [(self.root_folder_id, "")]

        while stack:
            folder_id, rel_prefix = stack.pop()
            for item in self._list_children(folder_id):
                name = item["name"]
                if item["mimeType"] == _FOLDER_MIME_TYPE:
                    if name in (CACHE_FOLDER_NAME, RESULTS_FOLDER_NAME):
                        if name == CACHE_FOLDER_NAME:
                            self._cache_folder_id = item["id"]
                        else:
                            self._results_folder_id = item["id"]
                        continue
                    stack.append((item["id"], f"{rel_prefix}{name}/"))
                elif item["mimeType"] in _IMAGE_MIME_TYPES:
                    self._image_map[f"{rel_prefix}{name}"] = item["id"]

        logger.info(f"Ditemukan {len(self._image_map)} gambar di folder Drive.")
        return self._image_map

    def download_folder_to_local(self, local_root: Path) -> Path:
        """
        Mengunduh seluruh gambar (dan cache '.face_cache' yang sudah ada di Drive, jika ada)
        ke sebuah folder lokal, mempertahankan struktur relative path.
        Berkas yang sudah ada secara lokal dilewati (sinkronisasi inkremental antar-run).

        Args:
            local_root (Path): Folder staging lokal tujuan unduhan.

        Returns:
            Path: local_root (untuk chaining).
        """
        local_root.mkdir(parents=True, exist_ok=True)
        self.list_images()

        image_jobs = []
        for rel_path, file_id in self._image_map.items():
            dst = local_root / rel_path
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            image_jobs.append((file_id, dst))

        if image_jobs:
            downloaded, failed = self._run_in_pool(
                image_jobs, self._download_job, lambda job: job[1].name, "drive-download",
            )
            msg = f"Mengunduh {downloaded} gambar baru dari Drive ke staging lokal."
            if failed:
                msg += f" ({len(failed)} gagal, akan dicoba lagi pada sinkronisasi berikutnya)"
            logger.info(msg)

        if self._cache_folder_id:
            cache_dir = local_root / CACHE_FOLDER_NAME
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_files = self._list_children(
                self._cache_folder_id, extra_query="mimeType = 'application/json'"
            )
            cache_jobs = []
            for item in cache_files:
                dst = cache_dir / item["name"]
                if dst.exists():
                    continue
                cache_jobs.append((item["id"], dst))

            if cache_jobs:
                downloaded, failed = self._run_in_pool(
                    cache_jobs, self._download_job, lambda job: job[1].name, "drive-download",
                )
                msg = f"Mengunduh {downloaded} berkas cache dari Drive."
                if failed:
                    msg += f" ({len(failed)} gagal)"
                logger.info(msg)

        logger.info(f"Folder Drive berhasil disinkronkan ke staging lokal: {local_root}")
        return local_root

    def _download_job(self, job: tuple):
        file_id, dst = job
        self._download_file(file_id, dst)

    def _download_file(self, file_id: str, dst: Path):
        """
        Mengunduh satu berkas dari Drive langsung ke disk secara streaming (chunk demi
        chunk sebesar DRIVE_DOWNLOAD_CHUNK_SIZE), tanpa menahan seluruh isi berkas di RAM
        sekaligus — penting karena dipanggil konkuren oleh beberapa worker thread, dan foto
        bisa berukuran beberapa MB per berkas (beda jauh dari cache JSON yang cuma puluhan KB).
        """
        request = self.service.files().get_media(fileId=file_id)
        request.http = self._thread_http()
        try:
            with open(dst, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request, chunksize=DRIVE_DOWNLOAD_CHUNK_SIZE)
                done = False
                while not done:
                    _, done = downloader.next_chunk(num_retries=_API_NUM_RETRIES)
        except Exception:
            # Jangan tinggalkan berkas parsial — kalau tidak, dst.exists() akan salah
            # mengira berkas ini sudah lengkap dan melewatkannya pada sinkronisasi berikutnya.
            dst.unlink(missing_ok=True)
            raise

    def upload_new_cache_files(self, local_root: Path):
        """
        Mengunggah berkas cache '.face_cache/*.json' yang baru dibuat secara lokal
        (belum ada di folder '.face_cache' Drive) kembali ke Drive.

        Args:
            local_root (Path): Folder staging lokal tempat proses indexing berjalan.
        """
        cache_dir = local_root / CACHE_FOLDER_NAME
        if not cache_dir.exists():
            return

        if not self._cache_folder_id:
            self._cache_folder_id = self._get_or_create_subfolder(self.root_folder_id, CACHE_FOLDER_NAME)

        existing_names = {
            item["name"]
            for item in self._list_children(self._cache_folder_id, extra_query="mimeType = 'application/json'")
        }

        pending = [f for f in cache_dir.glob("*.json") if f.name not in existing_names]
        if not pending:
            logger.info("Tidak ada berkas cache baru untuk diunggah ke Drive.")
            return

        uploaded, failed = self._run_in_pool(
            pending, self._upload_cache_file, lambda f: f.name, "drive-upload",
        )

        msg = f"Mengunggah {uploaded} berkas cache baru ke folder '.face_cache' di Drive."
        if failed:
            msg += f" ({len(failed)} gagal, akan dicoba lagi pada sinkronisasi berikutnya)"
        logger.info(msg)

    def _upload_cache_file(self, json_file: Path):
        metadata = {"name": json_file.name, "parents": [self._cache_folder_id]}
        media = MediaFileUpload(str(json_file), mimetype="application/json")
        self.service.files().create(
            body=metadata, media_body=media, fields="id",
        ).execute(http=self._thread_http(), num_retries=_API_NUM_RETRIES)

    def copy_matches_to_drive_results(self, relative_paths: list[str]) -> str | None:
        """
        Menyalin (server-side copy, tanpa unggah ulang byte gambar) seluruh foto hasil
        kecocokan ke subfolder 'Hasil_Pencarian_Selfie' di dalam folder Drive input.

        Args:
            relative_paths (list[str]): Relative path (relatif terhadap folder Drive input)
                dari foto-foto yang cocok, sesuai kunci pada peta list_images().

        Returns:
            str | None: ID folder hasil di Drive, atau None jika tidak ada yang disalin.
        """
        if not relative_paths:
            return None

        if not self._results_folder_id:
            self._results_folder_id = self._get_or_create_subfolder(self.root_folder_id, RESULTS_FOLDER_NAME)

        existing_names = {
            item["name"] for item in self._list_children(self._results_folder_id)
        }

        # Tentukan nama tujuan unik untuk tiap foto dulu, sekuensial dan murni lokal
        # (tanpa panggilan jaringan), sebelum menyalin secara paralel — supaya dua worker
        # thread tidak pernah berebut/bentrok memilih nama tujuan yang sama.
        jobs = []  # list of (file_id, dst_name)
        for rel_path in relative_paths:
            file_id = self._image_map.get(rel_path)
            if not file_id:
                logger.warning(f"Berkas '{rel_path}' tidak ditemukan di peta Drive, penyalinan dilewati.")
                continue

            base_name = Path(rel_path).name
            dst_name = base_name
            if dst_name in existing_names:
                stem, suffix = os.path.splitext(base_name)
                counter = 1
                while f"{stem}_{counter}{suffix}" in existing_names:
                    counter += 1
                dst_name = f"{stem}_{counter}{suffix}"

            existing_names.add(dst_name)
            jobs.append((file_id, dst_name))

        if not jobs:
            logger.info(f"Tidak ada foto hasil pencocokan yang disalin ke folder '{RESULTS_FOLDER_NAME}' di Drive.")
            return self._results_folder_id

        copied, failed = self._run_in_pool(
            jobs, self._copy_one_result, lambda job: job[1], "drive-copy",
        )

        msg = f"Menyalin {copied} foto hasil pencocokan ke folder '{RESULTS_FOLDER_NAME}' di Drive."
        if failed:
            msg += f" ({len(failed)} gagal)"
        logger.info(msg)
        return self._results_folder_id

    def _copy_one_result(self, job: tuple):
        file_id, dst_name = job
        self.service.files().copy(
            fileId=file_id,
            body={"name": dst_name, "parents": [self._results_folder_id]},
        ).execute(http=self._thread_http(), num_retries=_API_NUM_RETRIES)
