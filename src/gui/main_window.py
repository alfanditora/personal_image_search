import os
import gc
import time
import logging
import threading
import psutil
from concurrent.futures import ThreadPoolExecutor
from tkinter import filedialog, messagebox, Canvas
import customtkinter as ctk
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Claude-inspired warm light theme ───────────────────────────────────────────
BG_MAIN      = "#FAF9F5"   # warm ivory app background
BG_SIDEBAR   = "#F0EEE6"   # cream sidebar
BG_CARD      = "#FFFFFF"
BG_CARD_HV   = "#FAF9F5"
BG_INPUT     = "#FFFFFF"   # secondary buttons / entries (white on cream)
BG_INPUT_HV  = "#F5F4EF"
BG_THUMB     = "#F0EEE6"   # thumbnail placeholder inside white cards
BG_TRACK     = "#E6E4D9"   # progress track / segmented control track
C_ACCENT     = "#D97757"   # terracotta primary
C_ACCENT_HV  = "#C15F3C"
C_ACCENT_TX  = "#C15F3C"   # terracotta as text (darker for contrast)
C_INK        = "#29261B"   # near-black warm ink (Cari Foto button)
C_INK_HV     = "#3D3929"
C_SUCCESS    = "#6D8A66"   # muted sage green
C_SUCCESS_HV = "#5C7856"
C_ERROR      = "#BF4D43"
C_WARNING    = "#966C1E"
C_TEXT       = "#29261B"
C_SUBTEXT    = "#5E5D59"
C_MUTED      = "#83827D"
C_BORDER     = "#E0DED3"
C_BORDER_HV  = "#D3D1C5"
C_HDR_BG     = "#FAF9F5"

FONT_BODY  = "Segoe UI"
FONT_TITLE = "Georgia"     # serif title, Claude-style

FIXED_THRESHOLD = 0.60
THUMB_W, THUMB_H = 200, 140
THUMB_POOL_WORKERS = 4  # cap concurrent thumbnail decodes so large result sets don't blow up RAM
GALLERY_PAGE_SIZE = 60  # render results in pages so huge match counts don't spike widget/image RAM at once


class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Personal Image Search")
        self.geometry("1280x780")
        self.minsize(1000, 680)
        self.configure(fg_color=BG_MAIN)

        self.folder_path    = ""
        self.selfie_path    = ""
        self.input_mode     = "local"  # "local" | "drive"
        self.active_worker  = None
        self.detector       = None
        self.embedder       = None
        self.thumbnails     = []
        self._thumb_executor = ThreadPoolExecutor(max_workers=THUMB_POOL_WORKERS, thread_name_prefix="thumb")
        self._load_more_btn = None
        self._gallery_matches = []
        self._gallery_gt_basenames = None
        self._gallery_shown = 0

        # timer state
        self._timer_start   = 0.0
        self._timer_running = False
        self._elapsed       = 0.0

        # indexing timing breakdown (populated by _on_index_finished)
        self._index_timings = {}

        # memory footprint saat model detector/embedder pertama kali dimuat (RSS proses)
        self._model_init_memory_mb = 0.0

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_safe_close)

    # ══════════════════════════════════════════════════════════════════════════
    #  UI construction
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self._build_header()
        content = ctk.CTkFrame(self, fg_color=BG_MAIN, corner_radius=0)
        content.pack(fill="both", expand=True)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)
        self._build_sidebar(content)
        self._build_gallery(content)

    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=C_HDR_BG, height=62, corner_radius=0)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        # hairline under header
        ctk.CTkFrame(self, height=1, fg_color=C_BORDER, corner_radius=0).pack(fill="x", side="top")

        ctk.CTkLabel(
            hdr, text="Personal Image Search",
            font=ctk.CTkFont(FONT_TITLE, 19, "bold"),
            text_color=C_INK,
        ).pack(side="left", padx=24, pady=16)

        self.engine_badge = ctk.CTkLabel(
            hdr, text="●  SCRFD + ArcFace  |  OFFLINE",
            font=ctk.CTkFont(FONT_BODY, 11, "bold"),
            text_color=C_MUTED,
        )
        self.engine_badge.pack(side="right", padx=24)

        self._build_refresh_button(hdr).pack(side="right", padx=(0, 4))

    def _build_sidebar(self, parent):
        wrapper = ctk.CTkFrame(parent, width=300, fg_color=C_BORDER, corner_radius=0)
        wrapper.grid(row=0, column=0, sticky="nsew")
        wrapper.grid_propagate(False)
        wrapper.pack_propagate(False)

        sb = ctk.CTkScrollableFrame(
            wrapper, fg_color=BG_SIDEBAR, corner_radius=0,
            scrollbar_button_color=C_BORDER,
            scrollbar_button_hover_color=C_MUTED,
            scrollbar_fg_color=BG_SIDEBAR,
        )
        sb.pack(fill="both", expand=True, padx=(0, 1))
        sb.grid_columnconfigure(0, weight=1)

        r = 0

        # ── Folder ────────────────────────────────────────────────────────────
        self._section_label(sb, "INPUT FOLDER").grid(
            row=r, column=0, sticky="w", padx=16, pady=(24, 4)); r += 1

        self.input_mode_seg = ctk.CTkSegmentedButton(
            sb, values=["Folder Lokal", "Google Drive"],
            font=ctk.CTkFont(FONT_BODY, 11),
            fg_color=BG_TRACK,
            selected_color=BG_CARD, selected_hover_color=BG_CARD,
            unselected_color=BG_TRACK, unselected_hover_color="#DEDCD1",
            text_color=C_TEXT,
            command=self._on_input_mode_changed,
        )
        self.input_mode_seg.set("Folder Lokal")
        self.input_mode_seg.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 8)); r += 1

        # Kedua frame berikut menempati baris grid yang sama; hanya satu yang tampak sekaligus.
        input_row = r; r += 1

        self.local_input_frame = ctk.CTkFrame(sb, fg_color="transparent")
        self.local_input_frame.grid(row=input_row, column=0, sticky="ew", padx=0, pady=0)
        self.local_input_frame.grid_columnconfigure(0, weight=1)

        self._build_folder_card(self.local_input_frame).grid(
            row=0, column=0, sticky="ew", padx=16, pady=(0, 8))

        ctk.CTkButton(
            self.local_input_frame, text="Pilih Folder",
            font=ctk.CTkFont(FONT_BODY, 12),
            fg_color=BG_INPUT, hover_color=BG_INPUT_HV, text_color=C_TEXT,
            height=36, corner_radius=8, border_width=1, border_color=C_BORDER,
            command=self._select_folder,
        ).grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 6))

        self.drive_input_frame = ctk.CTkFrame(sb, fg_color="transparent")
        self.drive_input_frame.grid(row=input_row, column=0, sticky="ew", padx=0, pady=0)
        self.drive_input_frame.grid_columnconfigure(0, weight=1)

        self.drive_entry = ctk.CTkEntry(
            self.drive_input_frame, placeholder_text="Tempel tautan folder Google Drive...",
            font=ctk.CTkFont(FONT_BODY, 11),
            fg_color=BG_INPUT, border_color=C_BORDER, text_color=C_TEXT,
            placeholder_text_color=C_MUTED,
            height=36, corner_radius=8,
        )
        self.drive_entry.grid(row=0, column=0, sticky="ew", padx=16, pady=(0, 6))

        ctk.CTkLabel(
            self.drive_input_frame,
            text="Cache & hasil pencarian akan disimpan di folder Drive ini juga.",
            font=ctk.CTkFont(FONT_BODY, 10), text_color=C_MUTED,
            anchor="w", wraplength=716, justify="left",
        ).grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 6))

        self.drive_input_frame.grid_remove()  # tersembunyi selama mode lokal aktif

        self.btn_index = ctk.CTkButton(
            sb, text="Indeks Foto",
            font=ctk.CTkFont(FONT_BODY, 12, "bold"),
            fg_color=C_ACCENT, hover_color=C_ACCENT_HV, text_color="#FFFFFF",
            height=38, corner_radius=8,
            command=self._start_indexing,
        )
        self.btn_index.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 20)); r += 1

        self._divider(sb).grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 20)); r += 1

        # ── Selfie ────────────────────────────────────────────────────────────
        self._section_label(sb, "FOTO SELFIE").grid(
            row=r, column=0, sticky="w", padx=16, pady=(0, 4)); r += 1

        self.selfie_label = ctk.CTkLabel(
            sb, text="Belum dipilih",
            font=ctk.CTkFont(FONT_BODY, 11), text_color=C_MUTED,
            anchor="w", wraplength=716,
        )
        self.selfie_label.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 6)); r += 1

        ctk.CTkButton(
            sb, text="Pilih Selfie",
            font=ctk.CTkFont(FONT_BODY, 12),
            fg_color=BG_INPUT, hover_color=BG_INPUT_HV, text_color=C_TEXT,
            height=36, corner_radius=8, border_width=1, border_color=C_BORDER,
            command=self._select_selfie,
        ).grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 6)); r += 1

        self.btn_search = ctk.CTkButton(
            sb, text="Cari Foto",
            font=ctk.CTkFont(FONT_BODY, 13, "bold"),
            fg_color=C_INK, hover_color=C_INK_HV, text_color=BG_MAIN,
            height=42, corner_radius=8,
            command=self._start_search,
        )
        self.btn_search.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 20)); r += 1

        self._divider(sb).grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 20)); r += 1

        # ── Progress ──────────────────────────────────────────────────────────
        prow = ctk.CTkFrame(sb, fg_color=BG_SIDEBAR)
        prow.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 6)); r += 1
        prow.grid_columnconfigure(0, weight=1)
        self._section_label(prow, "PROGRES").grid(row=0, column=0, sticky="w")
        self.prog_pct = ctk.CTkLabel(
            prow, text="0%",
            font=ctk.CTkFont(FONT_BODY, 10, "bold"), text_color=C_ACCENT_TX,
        )
        self.prog_pct.grid(row=0, column=1, sticky="e")

        self.progress_bar = ctk.CTkProgressBar(
            sb, mode="determinate",
            progress_color=C_ACCENT, fg_color=BG_TRACK,
            height=6, corner_radius=3,
        )
        self.progress_bar.set(0)
        self.progress_bar.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 6)); r += 1

        self.status_label = ctk.CTkLabel(
            sb, text="Siap",
            font=ctk.CTkFont(FONT_BODY, 11), text_color=C_SUBTEXT,
            anchor="w", wraplength=716, fg_color=BG_SIDEBAR,
        )
        self.status_label.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 4)); r += 1

        # ── Time counter ──────────────────────────────────────────────────────
        self.time_label = ctk.CTkLabel(
            sb, text="",
            font=ctk.CTkFont(FONT_BODY, 10), text_color=C_MUTED,
            anchor="w", fg_color=BG_SIDEBAR,
        )
        self.time_label.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 24))

    def _build_gallery(self, parent):
        gal = ctk.CTkFrame(parent, fg_color=BG_MAIN, corner_radius=0)
        gal.grid(row=0, column=1, sticky="nsew")
        gal.grid_columnconfigure(0, weight=1)
        gal.grid_rowconfigure(1, weight=1)

        hdr_row = ctk.CTkFrame(gal, fg_color=BG_MAIN, height=52)
        hdr_row.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 0))
        hdr_row.grid_columnconfigure(0, weight=1)
        hdr_row.grid_propagate(False)

        ctk.CTkLabel(
            hdr_row, text="Hasil Pencarian",
            font=ctk.CTkFont(FONT_TITLE, 15, "bold"), text_color=C_INK, anchor="w",
        ).grid(row=0, column=0, sticky="w")

        self.result_count_label = ctk.CTkLabel(
            hdr_row, text="",
            font=ctk.CTkFont(FONT_BODY, 11), text_color=C_MUTED,
        )
        self.result_count_label.grid(row=0, column=1, sticky="e")

        self.gallery_scroll = ctk.CTkScrollableFrame(
            gal, fg_color=BG_MAIN,
            scrollbar_button_color=C_BORDER,
            scrollbar_button_hover_color=C_MUTED,
            scrollbar_fg_color=BG_MAIN,
        )
        self.gallery_scroll.grid(row=1, column=0, sticky="nsew", padx=(20, 8), pady=(8, 16))

        self._show_empty_state()

    # ══════════════════════════════════════════════════════════════════════════
    #  Widget helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _section_label(self, parent, text):
        return ctk.CTkLabel(
            parent, text=text,
            font=ctk.CTkFont(FONT_BODY, 10, "bold"), text_color=C_MUTED, anchor="w",
            fg_color="transparent",
        )

    def _divider(self, parent):
        return ctk.CTkFrame(parent, height=1, fg_color=C_BORDER, corner_radius=0)

    def _build_folder_card(self, parent) -> ctk.CTkFrame:
        """Kartu ikon yang menampilkan folder input aktif (bukan sekadar teks path polos)."""
        card = ctk.CTkFrame(
            parent, fg_color=BG_INPUT, border_width=1, border_color=C_BORDER, corner_radius=8,
        )
        card.grid_columnconfigure(1, weight=1)

        # Removed rowspan=2; icon sits cleanly in row 0
        icon_bg = ctk.CTkFrame(card, width=32, height=32, corner_radius=8, fg_color="#F5E4DE")
        icon_bg.grid(row=0, column=0, padx=(10, 9), pady=9)
        icon_bg.grid_propagate(False)

        icon = Canvas(icon_bg, width=16, height=16, bg="#F5E4DE", highlightthickness=0, bd=0)
        icon.place(relx=0.5, rely=0.5, anchor="center")
        icon.create_polygon(
            2, 5, 6, 5, 8, 7, 14, 7, 14, 13, 2, 13,
            outline=C_ACCENT_TX, fill="", width=1.4, joinstyle="round",
        )

        # Unified container for texts to eliminate the vertical grid stretching
        text_container = ctk.CTkFrame(card, fg_color="transparent")
        text_container.grid(row=0, column=1, sticky="w", padx=(0, 12), pady=9)

        self.folder_label = ctk.CTkLabel(
            text_container, text="Belum dipilih",
            font=ctk.CTkFont(FONT_BODY, 12, "bold"), text_color=C_MUTED, anchor="w",
            height=0 # Prevents the label from enforcing unnecessary default height
        )
        # Control the exact gap with pady=(0, 2)
        self.folder_label.pack(anchor="w", pady=(0, 2))

        ctk.CTkLabel(
            text_container, text="Folder Lokal",
            font=ctk.CTkFont(FONT_BODY, 10), text_color=C_MUTED, anchor="w",
            height=0
        ).pack(anchor="w", pady=0)

        return card

    def _set_status(self, text, color=None):
        self.status_label.configure(text=text, text_color=color or C_SUBTEXT)

    def _set_buttons_state(self, state: str):
        self.btn_index.configure(state=state)
        self.btn_search.configure(state=state)

    def _build_refresh_button(self, parent) -> ctk.CTkButton:
        return ctk.CTkButton(
            parent, text="Refresh",
            font=ctk.CTkFont(FONT_BODY, 11),
            fg_color=BG_INPUT, hover_color=BG_INPUT_HV, text_color=C_TEXT,
            height=32, width=88, corner_radius=8, border_width=1, border_color=C_BORDER,
            command=self._refresh_window,
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  Timer
    # ══════════════════════════════════════════════════════════════════════════

    def _start_timer(self):
        self._timer_start = time.time()
        self._timer_running = True
        self.time_label.configure(text="0.0s")
        self._tick_timer()

    def _stop_timer(self) -> float:
        self._timer_running = False
        self._elapsed = time.time() - self._timer_start
        mins, secs = divmod(self._elapsed, 60)
        text = f"{int(mins)}m {secs:.1f}s" if mins >= 1 else f"{self._elapsed:.1f}s"
        self.time_label.configure(text=f"Selesai dalam {text}")
        return self._elapsed

    def _tick_timer(self):
        if not self._timer_running:
            return
        elapsed = time.time() - self._timer_start
        mins, secs = divmod(elapsed, 60)
        text = f"{int(mins)}m {secs:.1f}s" if mins >= 1 else f"{elapsed:.1f}s"
        self.time_label.configure(text=text)
        self.after(100, self._tick_timer)

    # ══════════════════════════════════════════════════════════════════════════
    #  File selection
    # ══════════════════════════════════════════════════════════════════════════

    def _select_folder(self):
        path = filedialog.askdirectory(title="Pilih Folder Foto")
        if path:
            self.folder_path = path
            self.folder_label.configure(text=os.path.basename(path) or path, text_color=C_TEXT)

    def _on_input_mode_changed(self, value):
        if value == "Google Drive":
            self.input_mode = "drive"
            self.local_input_frame.grid_remove()
            self.drive_input_frame.grid()
        else:
            self.input_mode = "local"
            self.drive_input_frame.grid_remove()
            self.local_input_frame.grid()

    def _get_drive_link(self) -> str:
        return self.drive_entry.get().strip() if self.input_mode == "drive" else ""

    def _select_selfie(self):
        path = filedialog.askopenfilename(
            title="Pilih Foto Selfie",
            filetypes=[("Gambar", "*.jpg *.jpeg *.png"), ("Semua", "*.*")],
        )
        if path:
            self.selfie_path = path
            self.selfie_label.configure(text=os.path.basename(path), text_color=C_TEXT)

    # ══════════════════════════════════════════════════════════════════════════
    #  Model init (lazy)
    # ══════════════════════════════════════════════════════════════════════════

    def _ensure_models(self) -> bool:
        if self.detector is not None and self.embedder is not None:
            return True
        try:
            from core.detector import FaceDetector
            from core.embedder import ArcFaceEmbedder
            self._set_status("Memuat model SCRFD + ArcFace...", C_WARNING)
            self.update_idletasks()

            # Memory footprint inisialisasi pertama: RSS proses sebelum vs sesudah model dimuat
            proc = psutil.Process(os.getpid())
            mem_before = proc.memory_info().rss / (1024 ** 2)
            self.detector = FaceDetector("scrfd")
            self.embedder = ArcFaceEmbedder()
            mem_after = proc.memory_info().rss / (1024 ** 2)
            self._model_init_memory_mb = max(0.0, mem_after - mem_before)

            self.engine_badge.configure(text="●  SCRFD + ArcFace  |  ONLINE", text_color=C_SUCCESS)
            self._set_status("Model siap.", C_SUCCESS)
            return True
        except Exception as e:
            messagebox.showerror("Gagal Memuat Model", str(e))
            self._set_status("Gagal memuat model.", C_ERROR)
            return False

    # ══════════════════════════════════════════════════════════════════════════
    #  Indexing pipeline
    # ══════════════════════════════════════════════════════════════════════════

    def _start_indexing(self):
        drive_link = self._get_drive_link()
        if self.input_mode == "local" and not self.folder_path:
            messagebox.showwarning("Folder Belum Dipilih", "Pilih folder foto terlebih dahulu.")
            return
        if self.input_mode == "drive" and not drive_link:
            messagebox.showwarning("Tautan Drive Belum Diisi", "Tempel tautan folder Google Drive terlebih dahulu.")
            return
        if self.active_worker and self.active_worker.is_alive():
            messagebox.showinfo("Sedang Berjalan", "Proses masih berjalan.")
            return
        if not self._ensure_models():
            return

        self._set_buttons_state("disabled")
        self.progress_bar.set(0)
        self.prog_pct.configure(text="0%")
        self._set_status(
            "Memulai autentikasi & pengindeksan Google Drive..." if drive_link else "Memulai pengindeksan...",
            C_ACCENT_TX,
        )
        self._start_timer()

        from gui.workers import PipelineWorker
        self.active_worker = PipelineWorker(
            folder_path=self.folder_path,
            drive_link=drive_link or None,
            on_progress=self._on_index_progress,
            on_finished=self._on_index_finished,
            on_error=self._on_error,
            detector=self.detector,
            embedder=self.embedder,
        )
        self.active_worker.start()

    def _on_index_progress(self, current, total, filename):
        def _u():
            pct = current / total if total > 0 else 0
            self.progress_bar.set(pct)
            self.prog_pct.configure(text=f"{int(pct * 100)}%")
            short = filename if len(filename) <= 32 else "…" + filename[-30:]
            self._set_status(short)
        self.after(0, _u)

    def _on_index_finished(self, count, timings=None):
        self._index_timings = timings or {}
        self._index_timings["model_init_memory_mb"] = self._model_init_memory_mb
        def _u():
            self.progress_bar.set(1)
            self.prog_pct.configure(text="100%")
            self._stop_timer()
            self._set_status(f"Selesai — {count} foto terindeks.", C_SUCCESS)
            self._set_buttons_state("normal")
            messagebox.showinfo("Pengindeksan Selesai", f"{count} foto berhasil diindeks.")
        self.after(0, _u)

    # ══════════════════════════════════════════════════════════════════════════
    #  Search pipeline
    # ══════════════════════════════════════════════════════════════════════════

    def _start_search(self):
        drive_link = self._get_drive_link()
        if self.input_mode == "local" and not self.folder_path:
            messagebox.showwarning("Folder Belum Dipilih", "Pilih folder foto terlebih dahulu.")
            return
        if self.input_mode == "drive" and not drive_link:
            messagebox.showwarning("Tautan Drive Belum Diisi", "Tempel tautan folder Google Drive terlebih dahulu.")
            return
        if not self.selfie_path:
            messagebox.showwarning("Selfie Belum Dipilih", "Pilih foto selfie terlebih dahulu.")
            return
        if self.active_worker and self.active_worker.is_alive():
            messagebox.showinfo("Sedang Berjalan", "Proses masih berjalan.")
            return
        if not self._ensure_models():
            return

        self._set_buttons_state("disabled")
        self.progress_bar.set(0)
        self.prog_pct.configure(text="")
        self._set_status("Menganalisis selfie...", C_ACCENT_TX)
        self._show_empty_state()
        self.result_count_label.configure(text="")
        self.thumbnails.clear()
        self._start_timer()

        from gui.workers import SearchWorker
        self.active_worker = SearchWorker(
            folder_path=self.folder_path,
            drive_link=drive_link or None,
            query_img_path=self.selfie_path,
            threshold=FIXED_THRESHOLD,
            on_progress=self._on_search_progress,
            on_finished=self._on_search_finished,
            on_error=self._on_error,
            detector=self.detector,
            embedder=self.embedder,
        )
        self.active_worker.start()

    def _on_search_progress(self, message):
        def _u():
            self._set_status(message[:60] + "…" if len(message) > 60 else message)
        self.after(0, _u)

    def _on_search_finished(self, matches, output_folder):
        def _u():
            self.progress_bar.set(1)
            self._stop_timer()
            self._set_buttons_state("normal")
            if not matches:
                self._set_status("Tidak ada foto yang cocok.", C_WARNING)
                self._show_empty_state(no_results=True)
                self.result_count_label.configure(text="0 hasil")
                return
            self._set_status(f"Ditemukan {len(matches)} foto cocok.", C_SUCCESS)
            self.result_count_label.configure(text=f"{len(matches)} foto")
            self._display_gallery(matches)
        self.after(0, _u)

    # ══════════════════════════════════════════════════════════════════════════
    #  Error handler
    # ══════════════════════════════════════════════════════════════════════════

    def _on_error(self, msg):
        def _u():
            self._stop_timer()
            self._set_buttons_state("normal")
            self._set_status("Terjadi kesalahan.", C_ERROR)
            messagebox.showerror("Kesalahan", msg)
        self.after(0, _u)

    # ══════════════════════════════════════════════════════════════════════════
    #  Gallery rendering
    # ══════════════════════════════════════════════════════════════════════════

    def _clear_gallery(self):
        for w in self.gallery_scroll.winfo_children():
            w.destroy()
        self._load_more_btn = None
        self._gallery_matches = []
        self._gallery_gt_basenames = None
        self._gallery_shown = 0

    def _show_empty_state(self, no_results=False):
        self._clear_gallery()
        text = (
            "Tidak ada foto yang cocok\nuntuk selfie ini."
            if no_results
            else "Belum ada hasil pencarian.\n\nPilih folder & selfie,\nlalu klik  Cari Foto"
        )
        ctk.CTkLabel(
            self.gallery_scroll, text=f"\n{text}",
            font=ctk.CTkFont(FONT_BODY, 14), text_color=C_MUTED, justify="center",
        ).pack(expand=True, pady=80)

    def _display_gallery(self, matches: list, gt_basenames: set = None):
        # Render hasil secara bertahap (paginated) — membuat card + thumbnail untuk
        # ribuan hasil sekaligus adalah sumber utama overhead RAM setelah pencarian
        # selesai, jadi hanya GALLERY_PAGE_SIZE pertama yang dirender di awal.
        self._clear_gallery()
        self._gallery_matches = matches
        self._gallery_gt_basenames = gt_basenames
        COLS = 3
        for col in range(COLS):
            self.gallery_scroll.grid_columnconfigure(col, weight=1)
        self._render_more_results()

    def _render_more_results(self):
        matches = self._gallery_matches
        gt_basenames = self._gallery_gt_basenames
        COLS = 3
        start = self._gallery_shown
        end = min(start + GALLERY_PAGE_SIZE, len(matches))

        if self._load_more_btn is not None:
            self._load_more_btn.destroy()
            self._load_more_btn = None

        for idx in range(start, end):
            match = matches[idx]
            row, col = divmod(idx, COLS)
            basename = os.path.basename(match.get("file_name", ""))
            label = None
            if gt_basenames is not None:
                label = "TP" if basename in gt_basenames else "FP"
            card = self._make_card(match, result_label=label)
            card.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")

        self._gallery_shown = end
        if end < len(matches):
            remaining = len(matches) - end
            btn_row = (end - 1) // COLS + 1
            self._load_more_btn = ctk.CTkButton(
                self.gallery_scroll,
                text=f"Muat lebih banyak ({remaining} tersisa)",
                font=ctk.CTkFont(FONT_BODY, 12),
                fg_color=BG_INPUT, hover_color=BG_INPUT_HV, text_color=C_TEXT,
                border_width=1, border_color=C_BORDER, height=36,
                command=self._render_more_results,
            )
            self._load_more_btn.grid(row=btn_row, column=0, columnspan=COLS, pady=16, sticky="ew")

    def _make_card(self, match: dict, result_label: str = None) -> ctk.CTkFrame:
        file_path  = match.get("file_path", "")
        file_name  = match.get("file_name", os.path.basename(file_path))
        similarity = match.get("similarity", 0.0)
        distance   = match.get("cosine_distance", 1.0)

        card = ctk.CTkFrame(
            self.gallery_scroll, fg_color=BG_CARD, corner_radius=12,
            border_width=1, border_color=C_BORDER,
        )

        thumb_frame = ctk.CTkFrame(
            card, fg_color=BG_THUMB, corner_radius=8,
            width=THUMB_W, height=THUMB_H,
        )
        thumb_frame.pack(padx=10, pady=(10, 0))
        thumb_frame.pack_propagate(False)

        img_label = ctk.CTkLabel(thumb_frame, text="", fg_color="transparent")
        img_label.place(relx=0.5, rely=0.5, anchor="center")

        if result_label is not None:
            badge_color = C_SUCCESS if result_label == "TP" else C_ERROR
            ctk.CTkLabel(
                thumb_frame, text=result_label,
                font=ctk.CTkFont(FONT_BODY, 10, "bold"),
                text_color="#FFFFFF", fg_color=badge_color,
                corner_radius=4, width=28, height=18,
            ).place(relx=1.0, rely=0.0, anchor="ne", x=-6, y=6)

        display_name = file_name if len(file_name) <= 22 else file_name[:19] + "…"
        ctk.CTkLabel(
            card, text=display_name,
            font=ctk.CTkFont(FONT_BODY, 11, "bold"), text_color=C_TEXT,
            fg_color=BG_CARD,
        ).pack(pady=(8, 0))

        sim_pct   = similarity * 100
        sim_color = C_ACCENT_TX if sim_pct >= 65 else (C_WARNING if sim_pct >= 45 else C_SUBTEXT)
        ctk.CTkLabel(
            card, text=f"{sim_pct:.1f}% cocok",
            font=ctk.CTkFont(FONT_BODY, 11, "bold"), text_color=sim_color,
            fg_color=BG_CARD,
        ).pack()

        ctk.CTkLabel(
            card, text=f"jarak {distance:.3f}",
            font=ctk.CTkFont(FONT_BODY, 10), text_color=C_MUTED,
            fg_color=BG_CARD,
        ).pack(pady=(0, 10))

        def _open(e, fp=file_path):
            if os.path.exists(fp):
                os.startfile(fp)

        def _enter(e, c=card):
            c.configure(fg_color=BG_CARD_HV, border_color=C_BORDER_HV)

        def _leave(e, c=card):
            c.configure(fg_color=BG_CARD, border_color=C_BORDER)

        card.configure(cursor="hand2")
        for w in [card] + list(card.winfo_children()):
            try:
                w.configure(cursor="hand2")
            except Exception:
                pass
            w.bind("<Button-1>", _open)
            w.bind("<Enter>", _enter)
            w.bind("<Leave>", _leave)

        self._thumb_executor.submit(self._load_thumbnail, file_path, img_label)
        return card

    def _load_thumbnail(self, file_path: str, label: ctk.CTkLabel):
        try:
            img = Image.open(file_path)
            img.verify()
            img = Image.open(file_path)
            # Ask the JPEG decoder to decode at (roughly) the target size instead of full
            # resolution — avoids materializing a full-size decoded array per thumbnail,
            # which is what caused the RAM spike when a large result set is displayed.
            img.draft("RGB", (THUMB_W * 2, THUMB_H * 2))
            img = img.convert("RGB")

            iw, ih = img.size
            scale = max(THUMB_W / iw, THUMB_H / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            img = img.resize((nw, nh), Image.LANCZOS)
            left = (nw - THUMB_W) // 2
            top  = (nh - THUMB_H) // 2
            img  = img.crop((left, top, left + THUMB_W, top + THUMB_H))

            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(THUMB_W, THUMB_H))
            self.thumbnails.append(ctk_img)
            self.after(0, lambda lbl=label, ci=ctk_img: lbl.configure(image=ci, text=""))
        except Exception:
            self.after(0, lambda lbl=label: lbl.configure(text="⚠", text_color=C_ERROR))

    # ══════════════════════════════════════════════════════════════════════════
    #  Refresh / reset ke kondisi awal
    # ══════════════════════════════════════════════════════════════════════════

    def _refresh_window(self):
        """
        Menghentikan proses yang sedang berjalan (jika ada), mengembalikan seluruh
        tampilan ke kondisi awal, dan melepas model detector/embedder dari RAM
        (bukan hanya mereset UI) sehingga memori yang terpakai benar-benar dibebaskan.
        """
        if self.active_worker and self.active_worker.is_alive():
            if not messagebox.askyesno(
                "Proses Sedang Berjalan",
                "Ada proses yang masih berjalan. Hentikan proses tersebut dan reset tampilan ke kondisi awal?",
            ):
                return
            self.active_worker.stop()
            self.active_worker.join(timeout=2.0)

        self._timer_running = False
        self.active_worker = None

        # Reset input folder/selfie/drive
        self.folder_path = ""
        self.selfie_path = ""
        self.input_mode = "local"
        self.input_mode_seg.set("Folder Lokal")
        self._on_input_mode_changed("Folder Lokal")
        self.drive_entry.delete(0, "end")
        self.folder_label.configure(text="Belum dipilih", text_color=C_MUTED)
        self.selfie_label.configure(text="Belum dipilih", text_color=C_MUTED)

        # Reset progress, status, dan timer
        self.progress_bar.set(0)
        self.prog_pct.configure(text="0%")
        self.time_label.configure(text="")
        self._set_status("Siap", C_SUBTEXT)
        self._set_buttons_state("normal")

        # Reset galeri hasil pencarian
        self.thumbnails.clear()
        self.result_count_label.configure(text="")
        self._show_empty_state()

        # Lepaskan model dari RAM (sesi ONNX Runtime dkk.) — akan dimuat ulang otomatis
        # secara lazy oleh _ensure_models() saat pengguna memulai indexing/pencarian berikutnya
        self.detector = None
        self.embedder = None
        self.engine_badge.configure(text="●  SCRFD + ArcFace  |  OFFLINE", text_color=C_MUTED)

        self._index_timings = {}
        self._model_init_memory_mb = 0.0

        # Paksa garbage collection agar objek besar (model, embedding, thumbnail) yang
        # baru dilepas referensinya benar-benar dibebaskan oleh Python secepatnya
        gc.collect()

        logger.info("Jendela aplikasi di-refresh: state direset dan model dilepas dari RAM.")

    # ══════════════════════════════════════════════════════════════════════════
    #  Safe close
    # ══════════════════════════════════════════════════════════════════════════

    def on_safe_close(self):
        self._timer_running = False
        if self.active_worker and self.active_worker.is_alive():
            self.active_worker.stop()
        self._thumb_executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()
