import os
import time
import logging
import threading
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Light theme color tokens ───────────────────────────────────────────────────
BG_MAIN      = "#F8FAFC"
BG_SIDEBAR   = "#FFFFFF"
BG_CARD      = "#FFFFFF"
BG_CARD_HV   = "#F1F5F9"
BG_INPUT     = "#F1F5F9"
C_ACCENT     = "#2563EB"
C_ACCENT_HV  = "#1D4ED8"
C_SUCCESS    = "#10B981"
C_SUCCESS_HV = "#059669"
C_ERROR      = "#EF4444"
C_WARNING    = "#F59E0B"
C_TEXT       = "#0F172A"
C_SUBTEXT    = "#475569"
C_MUTED      = "#94A3B8"
C_BORDER     = "#E2E8F0"
C_BORDER_HV  = "#CBD5E1"
C_HDR_BG     = "#F1F5F9"

FIXED_THRESHOLD = 0.60
THUMB_W, THUMB_H = 200, 140


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

        # timer state
        self._timer_start   = 0.0
        self._timer_running = False
        self._elapsed       = 0.0

        # indexing timing breakdown (populated by _on_index_finished)
        self._index_timings = {}

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
        hdr = ctk.CTkFrame(self, fg_color=C_HDR_BG, height=60, corner_radius=0)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        ctk.CTkLabel(
            hdr, text="Personal Image Search",
            font=ctk.CTkFont("Segoe UI", 18, "bold"),
            text_color="#0F172A",
        ).pack(side="left", padx=24, pady=16)

        self.engine_badge = ctk.CTkLabel(
            hdr, text="● SCRFD + ArcFace  |  OFFLINE",
            font=ctk.CTkFont("Segoe UI", 11),
            text_color="#15803D",
        )
        self.engine_badge.pack(side="right", padx=24)

    def _build_sidebar(self, parent):
        wrapper = ctk.CTkFrame(parent, width=500, fg_color=C_BORDER, corner_radius=0)
        wrapper.grid(row=0, column=0, sticky="nsew")
        wrapper.grid_propagate(False)

        sb = ctk.CTkScrollableFrame(
            wrapper, fg_color=BG_SIDEBAR, corner_radius=0,
            scrollbar_button_color=C_BORDER,
            scrollbar_button_hover_color=C_MUTED,
            scrollbar_fg_color=BG_INPUT,
        )
        sb.pack(fill="both", expand=True, padx=(0, 1))
        sb.grid_columnconfigure(0, weight=1)

        r = 0

        # ── Folder ────────────────────────────────────────────────────────────
        self._section_label(sb, "INPUT FOLDER").grid(
            row=r, column=0, sticky="w", padx=16, pady=(24, 4)); r += 1

        self.input_mode_seg = ctk.CTkSegmentedButton(
            sb, values=["Folder Lokal", "Google Drive"],
            font=ctk.CTkFont("Segoe UI", 11),
            fg_color=BG_INPUT, selected_color=C_ACCENT, selected_hover_color=C_ACCENT_HV,
            unselected_color=BG_INPUT, unselected_hover_color=C_BORDER,
            command=self._on_input_mode_changed,
        )
        self.input_mode_seg.set("Folder Lokal")
        self.input_mode_seg.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 8)); r += 1

        # Kedua frame berikut menempati baris grid yang sama; hanya satu yang tampak sekaligus.
        input_row = r; r += 1

        self.local_input_frame = ctk.CTkFrame(sb, fg_color="transparent")
        self.local_input_frame.grid(row=input_row, column=0, sticky="ew", padx=0, pady=0)
        self.local_input_frame.grid_columnconfigure(0, weight=1)

        self.folder_label = ctk.CTkLabel(
            self.local_input_frame, text="Belum dipilih",
            font=ctk.CTkFont("Segoe UI", 11), text_color=C_MUTED,
            anchor="w", wraplength=456,
        )
        self.folder_label.grid(row=0, column=0, sticky="ew", padx=16, pady=(0, 6))

        ctk.CTkButton(
            self.local_input_frame, text="Pilih Folder",
            font=ctk.CTkFont("Segoe UI", 12),
            fg_color=BG_INPUT, hover_color=C_BORDER, text_color=C_TEXT,
            height=36, corner_radius=8, border_width=1, border_color=C_BORDER,
            command=self._select_folder,
        ).grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 6))

        self.drive_input_frame = ctk.CTkFrame(sb, fg_color="transparent")
        self.drive_input_frame.grid(row=input_row, column=0, sticky="ew", padx=0, pady=0)
        self.drive_input_frame.grid_columnconfigure(0, weight=1)

        self.drive_entry = ctk.CTkEntry(
            self.drive_input_frame, placeholder_text="Tempel tautan folder Google Drive...",
            font=ctk.CTkFont("Segoe UI", 11),
            fg_color=BG_INPUT, border_color=C_BORDER, text_color=C_TEXT,
            height=36, corner_radius=8,
        )
        self.drive_entry.grid(row=0, column=0, sticky="ew", padx=16, pady=(0, 6))

        ctk.CTkLabel(
            self.drive_input_frame,
            text="Cache & hasil pencarian akan disimpan di folder Drive ini juga.",
            font=ctk.CTkFont("Segoe UI", 10), text_color=C_MUTED,
            anchor="w", wraplength=456, justify="left",
        ).grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 6))

        self.drive_input_frame.grid_remove()  # tersembunyi selama mode lokal aktif

        self.btn_index = ctk.CTkButton(
            sb, text="Indeks Foto",
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
            fg_color=C_ACCENT, hover_color=C_ACCENT_HV, text_color="#FFFFFF",
            height=36, corner_radius=8,
            command=self._start_indexing,
        )
        self.btn_index.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 20)); r += 1

        self._divider(sb).grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 20)); r += 1

        # ── Selfie ────────────────────────────────────────────────────────────
        self._section_label(sb, "FOTO SELFIE").grid(
            row=r, column=0, sticky="w", padx=16, pady=(0, 4)); r += 1

        self.selfie_label = ctk.CTkLabel(
            sb, text="Belum dipilih",
            font=ctk.CTkFont("Segoe UI", 11), text_color=C_MUTED,
            anchor="w", wraplength=456,
        )
        self.selfie_label.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 6)); r += 1

        ctk.CTkButton(
            sb, text="Pilih Selfie",
            font=ctk.CTkFont("Segoe UI", 12),
            fg_color=BG_INPUT, hover_color=C_BORDER, text_color=C_TEXT,
            height=36, corner_radius=8, border_width=1, border_color=C_BORDER,
            command=self._select_selfie,
        ).grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 6)); r += 1

        self.btn_search = ctk.CTkButton(
            sb, text="Cari Foto",
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
            fg_color=C_SUCCESS, hover_color=C_SUCCESS_HV, text_color="#FFFFFF",
            height=40, corner_radius=8,
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
            font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color=C_ACCENT,
        )
        self.prog_pct.grid(row=0, column=1, sticky="e")

        self.progress_bar = ctk.CTkProgressBar(
            sb, mode="determinate",
            progress_color=C_ACCENT, fg_color=C_BORDER,
            height=6, corner_radius=3,
        )
        self.progress_bar.set(0)
        self.progress_bar.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 6)); r += 1

        self.status_label = ctk.CTkLabel(
            sb, text="Siap",
            font=ctk.CTkFont("Segoe UI", 11), text_color=C_SUBTEXT,
            anchor="w", wraplength=456, fg_color=BG_SIDEBAR,
        )
        self.status_label.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 4)); r += 1

        # ── Time counter ──────────────────────────────────────────────────────
        self.time_label = ctk.CTkLabel(
            sb, text="",
            font=ctk.CTkFont("Segoe UI", 10), text_color=C_MUTED,
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
            hdr_row, text="HASIL PENCARIAN",
            font=ctk.CTkFont("Segoe UI", 12, "bold"), text_color=C_SUBTEXT, anchor="w",
        ).grid(row=0, column=0, sticky="w")

        self.result_count_label = ctk.CTkLabel(
            hdr_row, text="",
            font=ctk.CTkFont("Segoe UI", 11), text_color=C_MUTED,
        )
        self.result_count_label.grid(row=0, column=1, sticky="e")

        self.gallery_scroll = ctk.CTkScrollableFrame(
            gal, fg_color=BG_MAIN,
            scrollbar_button_color=C_BORDER,
            scrollbar_button_hover_color=C_MUTED,
            scrollbar_fg_color=BG_INPUT,
        )
        self.gallery_scroll.grid(row=1, column=0, sticky="nsew", padx=(20, 8), pady=(8, 16))

        self._show_empty_state()

    # ══════════════════════════════════════════════════════════════════════════
    #  Widget helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _section_label(self, parent, text):
        return ctk.CTkLabel(
            parent, text=text,
            font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color=C_MUTED, anchor="w",
            fg_color="transparent",
        )

    def _divider(self, parent):
        return ctk.CTkFrame(parent, height=1, fg_color=C_BORDER, corner_radius=0)

    def _set_status(self, text, color=None):
        self.status_label.configure(text=text, text_color=color or C_SUBTEXT)

    def _set_buttons_state(self, state: str):
        self.btn_index.configure(state=state)
        self.btn_search.configure(state=state)

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
            self.detector = FaceDetector("scrfd")
            self.embedder = ArcFaceEmbedder()
            self.engine_badge.configure(text="● SCRFD + ArcFace  |  ONLINE", text_color="#4ADE80")
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
            C_ACCENT,
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
        self._set_status("Menganalisis selfie...", C_ACCENT)
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

    def _show_empty_state(self, no_results=False):
        self._clear_gallery()
        text = (
            "Tidak ada foto yang cocok\nuntuk selfie ini."
            if no_results
            else "Belum ada hasil pencarian.\n\nPilih folder & selfie,\nlalu klik  Cari Foto"
        )
        ctk.CTkLabel(
            self.gallery_scroll, text=f"\n{text}",
            font=ctk.CTkFont("Segoe UI", 14), text_color=C_MUTED, justify="center",
        ).pack(expand=True, pady=80)

    def _display_gallery(self, matches: list, gt_basenames: set = None):
        self._clear_gallery()
        COLS = 3
        for col in range(COLS):
            self.gallery_scroll.grid_columnconfigure(col, weight=1)
        for idx, match in enumerate(matches):
            row, col = divmod(idx, COLS)
            basename = os.path.basename(match.get("file_name", ""))
            label = None
            if gt_basenames is not None:
                label = "TP" if basename in gt_basenames else "FP"
            card = self._make_card(match, result_label=label)
            card.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")

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
            card, fg_color=BG_INPUT, corner_radius=8,
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
                font=ctk.CTkFont("Segoe UI", 10, "bold"),
                text_color="#FFFFFF", fg_color=badge_color,
                corner_radius=4, width=28, height=18,
            ).place(relx=1.0, rely=0.0, anchor="ne", x=-6, y=6)

        display_name = file_name if len(file_name) <= 22 else file_name[:19] + "…"
        ctk.CTkLabel(
            card, text=display_name,
            font=ctk.CTkFont("Segoe UI", 11, "bold"), text_color=C_TEXT,
            fg_color=BG_CARD,
        ).pack(pady=(8, 0))

        sim_pct   = similarity * 100
        sim_color = C_SUCCESS if sim_pct >= 65 else (C_WARNING if sim_pct >= 45 else C_SUBTEXT)
        ctk.CTkLabel(
            card, text=f"{sim_pct:.1f}% cocok",
            font=ctk.CTkFont("Segoe UI", 11), text_color=sim_color,
            fg_color=BG_CARD,
        ).pack()

        ctk.CTkLabel(
            card, text=f"jarak {distance:.3f}",
            font=ctk.CTkFont("Segoe UI", 10), text_color=C_MUTED,
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

        threading.Thread(
            target=self._load_thumbnail, args=(file_path, img_label), daemon=True,
        ).start()
        return card

    def _load_thumbnail(self, file_path: str, label: ctk.CTkLabel):
        try:
            img = Image.open(file_path)
            img.verify()
            img = Image.open(file_path).convert("RGB")

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
    #  Safe close
    # ══════════════════════════════════════════════════════════════════════════

    def on_safe_close(self):
        self._timer_running = False
        if self.active_worker and self.active_worker.is_alive():
            self.active_worker.stop()
        self.destroy()
