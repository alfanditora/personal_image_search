"""
Demo GUI — extends the production GUI with:
  • Ground truth file picker (JSON)
  • TP/FP badge on each result card
  • Metrics report panel (Precision, Recall, F1, Accuracy + confusion counts + time)

Ground truth JSON format (two accepted variants):
  Flat list:   ["img1.jpg", "img5.jpg", ...]
  Flat dict:   {"img1.jpg": true, "img2.jpg": false}
  Multi-person: {"img1.jpg": {"contains_person_A": true, "contains_person_B": false, ...}, ...}

For multi-person format the user selects which person (A/B/C) to evaluate against.
TN and Accuracy always use the GT image count as the denominator (exact; no folder scan needed).
"""

import os
import json
import logging
from tkinter import filedialog, messagebox, Canvas
import customtkinter as ctk

from gui.main_window import (
    MainApp,
    BG_MAIN, BG_SIDEBAR, BG_CARD, BG_CARD_HV, BG_INPUT, BG_INPUT_HV, BG_TRACK,
    C_ACCENT, C_ACCENT_HV, C_ACCENT_TX, C_INK, C_INK_HV, C_SUCCESS, C_SUCCESS_HV,
    C_ERROR, C_WARNING, C_TEXT, C_SUBTEXT, C_MUTED,
    C_BORDER, C_HDR_BG, FONT_BODY, FONT_TITLE,
    THUMB_W, THUMB_H,
)

logger = logging.getLogger(__name__)

# ── Report panel — warm-neutral tints (Claude-inspired palette) ────────────────
C_REPORT_BG   = "#F5F1E8"
C_TP_BG, C_TP_FG = "#E7EFE2", "#4B6B45"
C_FP_BG, C_FP_FG = "#F5E1DA", "#9C4A3A"
C_FN_BG, C_FN_FG = "#F7EEDA", "#8A6A22"
C_TN_BG, C_TN_FG = "#ECEAE0", "#6B6960"
C_DETECT_BG, C_DETECT_FG = "#E8ECF2", "#40597A"

# Maps display label → JSON key used in multi-person GT
PERSON_KEY_MAP = {
    "A": "contains_person_A",
    "B": "contains_person_B",
    "C": "contains_person_C",
}


class DemoApp(MainApp):
    """Face search demo with ground truth evaluation report."""

    def __init__(self):
        self._gt_path           = ""
        self._ground_truth      = None   # flat format: {basename: bool}
        self._ground_truth_raw  = None   # multi-person: {basename: {person_key: bool}}
        self._report_frame      = None
        self._report_body       = None
        self._report_collapsed  = False
        self._query_person      = None   # StringVar — created in _build_sidebar after Tk root
        super().__init__()
        self.title("Personal Image Search — Demo")

    # ── Override header ───────────────────────────────────────────────────────

    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=C_HDR_BG, height=60, corner_radius=0)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        ctk.CTkFrame(self, height=1, fg_color=C_BORDER, corner_radius=0).pack(fill="x", side="top")

        title_frame = ctk.CTkFrame(hdr, fg_color="transparent")
        title_frame.pack(side="left", padx=24, pady=10)

        ctk.CTkLabel(
            title_frame, text="Personal Image Search",
            font=ctk.CTkFont(FONT_TITLE, 19, "bold"),
            text_color="#29261B",
        ).pack(side="left")

        ctk.CTkLabel(
            title_frame, text="  DEMO",
            font=ctk.CTkFont(FONT_BODY, 11, "bold"),
            text_color=C_ACCENT_TX,
            fg_color="#F5E4DE",
            corner_radius=4,
        ).pack(side="left", padx=(8, 0))

        self.engine_badge = ctk.CTkLabel(
            hdr, text="●  SCRFD + ArcFace  |  OFFLINE",
            font=ctk.CTkFont(FONT_BODY, 11, "bold"),
            text_color=C_MUTED,
        )
        self.engine_badge.pack(side="right", padx=24)

        self._build_refresh_button(hdr).pack(side="right", padx=(0, 4))

    # ── Override sidebar — Ground Truth + Query Person + Threshold slider ─────

    def _build_sidebar(self, parent):
        # DoubleVar and StringVar must be created after the Tk root exists
        self._threshold    = ctk.DoubleVar(value=0.60)
        self._query_person = ctk.StringVar(value="A")

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
            height=40, corner_radius=8,
            command=self._start_search,
        )
        self.btn_search.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 20)); r += 1

        self._divider(sb).grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 20)); r += 1

        # ── Ground Truth ──────────────────────────────────────────────────────
        self._section_label(sb, "GROUND TRUTH").grid(
            row=r, column=0, sticky="w", padx=16, pady=(0, 4)); r += 1

        self._gt_status_label = ctk.CTkLabel(
            sb, text="Belum dimuat",
            font=ctk.CTkFont(FONT_BODY, 11), text_color=C_MUTED,
            anchor="w", wraplength=716, fg_color=BG_SIDEBAR,
        )
        self._gt_status_label.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 6)); r += 1

        ctk.CTkButton(
            sb, text="Pilih File Ground Truth",
            font=ctk.CTkFont(FONT_BODY, 12),
            fg_color=BG_INPUT, hover_color=BG_INPUT_HV, text_color=C_TEXT,
            height=36, corner_radius=8, border_width=1, border_color=C_BORDER,
            command=self._select_ground_truth,
        ).grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 12)); r += 1

        # ── Query Person (active only for multi-person GT) ────────────────────
        self._section_label(sb, "QUERY PERSON").grid(
            row=r, column=0, sticky="w", padx=16, pady=(0, 4)); r += 1

        self._person_btn = ctk.CTkSegmentedButton(
            sb,
            values=["A", "B", "C"],
            variable=self._query_person,
            fg_color=BG_TRACK,
            selected_color=C_ACCENT,
            selected_hover_color=C_ACCENT_HV,
            unselected_color=BG_TRACK,
            unselected_hover_color="#DEDCD1",
            text_color=C_TEXT,
            font=ctk.CTkFont(FONT_BODY, 12, "bold"),
            state="disabled",
        )
        self._person_btn.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 20)); r += 1

        self._divider(sb).grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 20)); r += 1

        # ── Threshold slider (demo only) ──────────────────────────────────────
        trow = ctk.CTkFrame(sb, fg_color=BG_SIDEBAR)
        trow.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 6)); r += 1
        trow.grid_columnconfigure(0, weight=1)
        self._section_label(trow, "THRESHOLD KEMIRIPAN").grid(row=0, column=0, sticky="w")
        self.thresh_val_label = ctk.CTkLabel(
            trow, text="0.60",
            font=ctk.CTkFont(FONT_BODY, 12, "bold"), text_color=C_ACCENT_TX,
            fg_color=BG_SIDEBAR,
        )
        self.thresh_val_label.grid(row=0, column=1, sticky="e")

        ctk.CTkSlider(
            sb, from_=0.10, to=1.00, number_of_steps=18,
            variable=self._threshold,
            fg_color=BG_TRACK,
            button_color=C_ACCENT, button_hover_color=C_ACCENT_HV,
            progress_color=C_ACCENT,
            command=self._on_threshold_change,
        ).grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 20)); r += 1

        self._divider(sb).grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 20)); r += 1

        # ── Progress ──────────────────────────────────────────────────────────
        prow = ctk.CTkFrame(sb, fg_color=BG_SIDEBAR)
        prow.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 6)); r += 1
        prow.grid_columnconfigure(0, weight=1)
        self._section_label(prow, "PROGRES").grid(row=0, column=0, sticky="w")
        self.prog_pct = ctk.CTkLabel(
            prow, text="0%",
            font=ctk.CTkFont(FONT_BODY, 10, "bold"), text_color=C_ACCENT_TX,
            fg_color=BG_SIDEBAR,
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

        self.time_label = ctk.CTkLabel(
            sb, text="",
            font=ctk.CTkFont(FONT_BODY, 10), text_color=C_MUTED,
            anchor="w", fg_color=BG_SIDEBAR,
        )
        self.time_label.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 24))

    # ── Override gallery to include report panel row ──────────────────────────

    def _build_gallery(self, parent):
        self._gal_frame = ctk.CTkFrame(parent, fg_color=BG_MAIN, corner_radius=0)
        self._gal_frame.grid(row=0, column=1, sticky="nsew")
        self._gal_frame.grid_columnconfigure(0, weight=1)
        self._gal_frame.grid_rowconfigure(3, weight=1)

        hdr_row = ctk.CTkFrame(self._gal_frame, fg_color=BG_MAIN, height=52)
        hdr_row.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 0))
        hdr_row.grid_columnconfigure(0, weight=1)
        hdr_row.grid_propagate(False)

        ctk.CTkLabel(
            hdr_row, text="Hasil Pencarian",
            font=ctk.CTkFont(FONT_TITLE, 15, "bold"), text_color="#29261B", anchor="w",
        ).grid(row=0, column=0, sticky="w")

        self.result_count_label = ctk.CTkLabel(
            hdr_row, text="",
            font=ctk.CTkFont(FONT_BODY, 11), text_color=C_MUTED,
        )
        self.result_count_label.grid(row=0, column=1, sticky="e")

        # ── Filter kategori (Semua / TP / FP / FN) — supaya tidak perlu scroll
        # untuk mencari kartu FP/FN di antara ratusan kartu TP ────────────────
        self._gallery_filter_var = ctk.StringVar(value="Semua")
        filter_row = ctk.CTkFrame(self._gal_frame, fg_color=BG_MAIN)
        filter_row.grid(row=1, column=0, sticky="ew", padx=20, pady=(10, 0))

        self._gallery_filter_seg = ctk.CTkSegmentedButton(
            filter_row,
            values=["Semua", "TP", "FP", "FN"],
            variable=self._gallery_filter_var,
            font=ctk.CTkFont(FONT_BODY, 11, "bold"),
            fg_color=BG_TRACK,
            selected_color=C_ACCENT, selected_hover_color=C_ACCENT_HV,
            unselected_color=BG_TRACK, unselected_hover_color="#DEDCD1",
            text_color=C_TEXT,
            command=self._on_gallery_filter_changed,
        )
        self._gallery_filter_seg.pack(side="left")

        # Report panel (row=2, hidden until metrics are ready)
        self._report_frame = ctk.CTkFrame(
            self._gal_frame, fg_color=BG_CARD, corner_radius=12,
            border_width=1, border_color=C_BORDER,
        )

        self.gallery_scroll = ctk.CTkScrollableFrame(
            self._gal_frame, fg_color=BG_MAIN,
            scrollbar_button_color=C_BORDER,
            scrollbar_button_hover_color=C_MUTED,
            scrollbar_fg_color=BG_MAIN,
        )
        self.gallery_scroll.grid(row=3, column=0, sticky="nsew", padx=(20, 8), pady=(8, 16))

        self._show_empty_state()

    # ── Gallery filter callback ────────────────────────────────────────────────

    _FILTER_LABEL_TO_KEY = {"Semua": "ALL", "TP": "TP", "FP": "FP", "FN": "FN"}

    def _on_gallery_filter_changed(self, value):
        self._set_gallery_filter(self._FILTER_LABEL_TO_KEY.get(value, "ALL"))

    def _sync_gallery_filter_ui(self):
        # Dipanggil oleh MainApp._display_gallery setiap ada hasil pencarian baru,
        # supaya filter kembali ke "Semua" alih-alih menyisakan pilihan lama.
        if hasattr(self, "_gallery_filter_var"):
            self._gallery_filter_var.set("Semua")

    # ── Threshold callback ────────────────────────────────────────────────────

    def _on_threshold_change(self, value):
        rounded = round(value / 0.05) * 0.05
        self.thresh_val_label.configure(text=f"{rounded:.2f}")

    # ── Override refresh — juga reset Ground Truth, Query Person, Threshold, Report ──

    def _refresh_window(self):
        super()._refresh_window()

        self._gt_path = ""
        self._ground_truth = None
        self._ground_truth_raw = None
        self._gt_status_label.configure(text="Belum dimuat", text_color=C_MUTED)

        self._query_person.set("A")
        self._person_btn.configure(state="disabled")

        self._threshold.set(0.60)
        self.thresh_val_label.configure(text="0.60")

        self._gallery_filter = None
        self._sync_gallery_filter_ui()

        self._hide_report()

    # ── Override search to read slider threshold ──────────────────────────────

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
        self._hide_report()
        self._start_timer()

        threshold = round(self._threshold.get() / 0.05) * 0.05

        from gui.workers import SearchWorker
        self.active_worker = SearchWorker(
            folder_path=self.folder_path,
            drive_link=drive_link or None,
            query_img_path=self.selfie_path,
            threshold=threshold,
            on_progress=self._on_search_progress,
            on_finished=self._on_search_finished,
            on_error=self._on_error,
            detector=self.detector,
            embedder=self.embedder,
        )
        self.active_worker.start()

    # ── Ground truth loading ──────────────────────────────────────────────────

    def _select_ground_truth(self):
        path = filedialog.askopenfilename(
            title="Pilih File Ground Truth",
            filetypes=[("JSON", "*.json"), ("Semua", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            fname = os.path.basename(path)

            if isinstance(raw, list):
                # Flat list of positive filenames
                self._ground_truth = {os.path.basename(name): True for name in raw}
                self._ground_truth_raw = None
                pos = len(self._ground_truth)
                self._gt_status_label.configure(
                    text=f"{fname}\n{pos} positif (format list)",
                    text_color=C_SUCCESS,
                )
                self._person_btn.configure(state="disabled")

            elif isinstance(raw, dict) and raw:
                first_val = next(iter(raw.values()))

                if isinstance(first_val, dict):
                    # Multi-person format: {filename: {contains_person_X: bool}}
                    self._ground_truth_raw = {os.path.basename(k): v for k, v in raw.items()}
                    self._ground_truth = None

                    # Count positives per person for the status label
                    person_keys = list(next(iter(self._ground_truth_raw.values())).keys())
                    counts = {}
                    for key in person_keys:
                        counts[key] = sum(
                            1 for labels in self._ground_truth_raw.values()
                            if labels.get(key) is True
                        )
                    # Build compact summary: "A: 456 | B: 154 | C: 97"
                    summary_parts = []
                    for key in person_keys:
                        label = key.replace("contains_person_", "")
                        summary_parts.append(f"{label}: {counts[key]}")
                    summary = "  |  ".join(summary_parts)

                    self._gt_status_label.configure(
                        text=f"{fname}\n{len(self._ground_truth_raw)} gambar  ·  {summary}",
                        text_color=C_SUCCESS,
                    )
                    self._person_btn.configure(state="normal")

                else:
                    # Flat dict: {filename: bool}
                    self._ground_truth = {os.path.basename(k): bool(v) for k, v in raw.items()}
                    self._ground_truth_raw = None
                    pos = sum(1 for v in self._ground_truth.values() if v)
                    self._gt_status_label.configure(
                        text=f"{fname}\n{pos} positif dari {len(self._ground_truth)} entri",
                        text_color=C_SUCCESS,
                    )
                    self._person_btn.configure(state="disabled")
            else:
                raise ValueError("Format tidak dikenali. Gunakan list, flat dict, atau multi-person dict.")

            self._gt_path = path

        except Exception as e:
            messagebox.showerror("Gagal Memuat Ground Truth", str(e))
            self._gt_status_label.configure(text="Gagal dimuat", text_color=C_ERROR)

    # ── Override search finished ──────────────────────────────────────────────

    def _on_search_finished(self, matches, output_folder):
        def _u():
            self.progress_bar.set(1)
            self._stop_timer()
            self._set_buttons_state("normal")

            gt_basenames = None
            if self._ground_truth_raw:
                # Multi-person GT: extract column for the selected person
                person_key = PERSON_KEY_MAP.get(
                    self._query_person.get(), "contains_person_A"
                )
                gt_basenames = {
                    fname for fname, labels in self._ground_truth_raw.items()
                    if labels.get(person_key) is True
                }
            elif self._ground_truth:
                # Flat GT
                gt_basenames = {k for k, v in self._ground_truth.items() if v}

            fn_matches = []
            if gt_basenames is not None:
                fn_matches = self._build_fn_matches(matches, gt_basenames)
                self._compute_and_show_report(matches, gt_basenames)
            else:
                self._hide_report()

            total_shown = len(matches) + len(fn_matches)

            if total_shown == 0:
                self._set_status("Tidak ada foto yang cocok.", C_WARNING)
                self._show_empty_state(no_results=True)
                self.result_count_label.configure(text="0 hasil")
                return

            if matches:
                self._set_status(f"Ditemukan {len(matches)} foto cocok.", C_SUCCESS)
            else:
                self._set_status("Tidak ada foto cocok ditemukan; kartu FN ditampilkan.", C_WARNING)

            self.result_count_label.configure(
                text=f"{len(matches)} foto  ·  {len(fn_matches)} FN"
                if gt_basenames is not None else f"{len(matches)} foto"
            )

            self._display_gallery(matches, gt_basenames=gt_basenames, fn_matches=fn_matches)
        self.after(0, _u)

    # ── False Negative reconstruction (demo only) ──────────────────────────────

    def _build_fn_matches(self, matches: list, gt_positive: set) -> list:
        """Foto ground truth positif yang gagal ditemukan pencarian (di bawah TP).
        Direkonstruksi dari cache indexing (atau, jika wajahnya tak pernah
        terdeteksi sama sekali, dari pemindaian folder langsung) supaya tetap
        bisa ditampilkan sebagai kartu FN di galeri demo."""
        matched_basenames = {os.path.basename(m.get("file_name", "")) for m in matches}
        fn_basenames = gt_positive - matched_basenames
        if not fn_basenames:
            return []

        worker = self.active_worker
        cache_list = getattr(worker, "cache_list", None) or []
        effective_folder = getattr(worker, "effective_folder", None) or self.folder_path
        is_drive = getattr(worker, "is_drive", False)
        selfie_embedding = getattr(worker, "selfie_embedding", None)

        cache_by_basename = {}
        for item in cache_list:
            b = os.path.basename(item.get("file_name", ""))
            if b in fn_basenames and b not in cache_by_basename:
                cache_by_basename[b] = item

        from core.matcher import FaceMatcher
        matcher = FaceMatcher()

        fn_matches = []
        for basename in fn_basenames:
            item = cache_by_basename.get(basename)
            file_path = None
            distance, similarity = 1.0, 0.0

            if item is not None:
                rel_name = item.get("file_name", basename)
                file_path = (
                    os.path.join(effective_folder, rel_name) if is_drive
                    else item.get("file_path") or os.path.join(effective_folder, rel_name)
                )
                emb = item.get("embedding")
                if selfie_embedding is not None and emb is not None:
                    distance = matcher.hitung_kesamaan_cosine(selfie_embedding, emb)
                    similarity = 1.0 - distance
            else:
                # Wajah tak pernah terdeteksi saat indexing (mis. sudut/pencahayaan
                # buruk) — cari langsung berdasarkan nama berkas di folder.
                file_path = self._locate_file_by_basename(effective_folder, basename)

            if file_path and os.path.exists(file_path):
                fn_matches.append({
                    "file_name": basename,
                    "file_path": file_path,
                    "similarity": similarity,
                    "cosine_distance": distance,
                })
        return fn_matches

    @staticmethod
    def _locate_file_by_basename(root_folder: str, basename: str):
        if not root_folder or not os.path.isdir(root_folder):
            return None
        for dirpath, dirnames, files in os.walk(root_folder):
            dirnames[:] = [d for d in dirnames if d not in (".face_cache", "Hasil_Pencarian_Selfie")]
            if basename in files:
                return os.path.join(dirpath, basename)
        return None

    # ── Report panel ──────────────────────────────────────────────────────────

    def _hide_report(self):
        self._report_frame.grid_forget()

    def _compute_and_show_report(self, matches: list, gt_positive: set):
        matched_basenames = {os.path.basename(m.get("file_name", "")) for m in matches}

        tp = len(matched_basenames & gt_positive)
        fp = len(matched_basenames - gt_positive)
        fn = len(gt_positive - matched_basenames)

        # Use GT size as the authoritative total (covers entire dataset)
        if self._ground_truth_raw:
            total_images = len(self._ground_truth_raw)
        elif self._ground_truth:
            total_images = len(self._ground_truth)
        else:
            total_images = tp + fp + fn  # fallback: no TN info

        tn = max(0, total_images - tp - fp - fn)
        accuracy = (tp + tn) / total_images if total_images > 0 else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        # Report title shows which person was queried (for multi-person GT)
        person_label = None
        if self._ground_truth_raw:
            person_label = f"Person {self._query_person.get()}"

        # Jumlah wajah yang berhasil dideteksi oleh face detection saat proses indexing terakhir
        faces_detected = self._index_timings.get("faces_detected")
        # Total waktu sinkronisasi Google Drive (indexing + search), 0 jika mode lokal
        drive_sync_time = (
            self._index_timings.get("drive_sync_time", 0.0)
            + getattr(self.active_worker, "drive_sync_time", 0.0)
        )
        # Memory footprint saat model detector/embedder pertama kali dimuat (RSS proses)
        model_init_memory_mb = self._index_timings.get("model_init_memory_mb")
        # Memory incremental: pertambahan RSS proses selama loop pengindeksan terakhir berjalan
        # (deteksi + alignment + embedding seluruh foto baru), terpisah dari footprint inisialisasi model
        memory_incremental_mb = self._index_timings.get("memory_incremental_mb")
        # Penggunaan CPU proses selama proses pengindeksan terakhir berjalan
        cpu_usage_percent = self._index_timings.get("cpu_usage_percent")

        self._render_report(tp, fp, fn, tn, precision, recall, f1, accuracy,
                            person_label=person_label,
                            total_images=total_images,
                            positive_count=len(gt_positive),
                            faces_detected=faces_detected,
                            drive_sync_time=drive_sync_time,
                            model_init_memory_mb=model_init_memory_mb,
                            memory_incremental_mb=memory_incremental_mb,
                            cpu_usage_percent=cpu_usage_percent)

    def _render_report(self, tp, fp, fn, tn, precision, recall, f1, accuracy,
                       person_label=None, total_images=None, positive_count=None,
                       faces_detected=None, drive_sync_time=0.0,
                       model_init_memory_mb=None, memory_incremental_mb=None,
                       cpu_usage_percent=None):
        for w in self._report_frame.winfo_children():
            w.destroy()

        # ── Header: icon + title + person pill ────────────────────────────────
        hdr_row = ctk.CTkFrame(self._report_frame, fg_color=BG_CARD)
        hdr_row.pack(fill="x", padx=20, pady=(18, 14))
        hdr_row.grid_columnconfigure(1, weight=1)

        icon_bg = ctk.CTkFrame(hdr_row, width=36, height=36, corner_radius=9, fg_color="#F5E4DE")
        icon_bg.grid(row=0, column=0, rowspan=2, padx=(0, 12))
        icon_bg.grid_propagate(False)
        icon = Canvas(icon_bg, width=18, height=18, bg="#F5E4DE", highlightthickness=0, bd=0)
        icon.place(relx=0.5, rely=0.5, anchor="center")
        icon.create_line(3, 2, 3, 16, 17, 16, fill=C_ACCENT_TX, width=1.6, capstyle="round", joinstyle="round")
        icon.create_line(6, 13, 9, 8, 12, 10, 16, 4, fill=C_ACCENT_TX, width=1.6, capstyle="round", joinstyle="round")

        ctk.CTkLabel(
            hdr_row, text="Laporan Evaluasi",
            font=ctk.CTkFont(FONT_TITLE, 15, "bold"), text_color=C_TEXT,
            fg_color=BG_CARD, anchor="w",
        ).grid(row=0, column=1, sticky="w")

        subtitle = ""
        if total_images is not None and positive_count is not None:
            subtitle = f"{positive_count} positif dari {total_images} gambar diverifikasi"
        ctk.CTkLabel(
            hdr_row, text=subtitle,
            font=ctk.CTkFont(FONT_BODY, 11), text_color=C_MUTED,
            fg_color=BG_CARD, anchor="w",
        ).grid(row=1, column=1, sticky="w")

        if person_label:
            ctk.CTkLabel(
                hdr_row, text=f"  {person_label}  ",
                font=ctk.CTkFont(FONT_BODY, 11, "bold"), text_color=C_ACCENT_TX,
                fg_color="#F5E4DE", corner_radius=20,
            ).grid(row=0, column=2, rowspan=2, sticky="e", padx=(12, 0))

        self._report_toggle_btn = ctk.CTkButton(
            hdr_row, text="▲ Sembunyikan Detail",
            font=ctk.CTkFont(FONT_BODY, 10, "bold"),
            fg_color=BG_INPUT, hover_color=BG_INPUT_HV, text_color=C_SUBTEXT,
            border_width=1, border_color=C_BORDER,
            width=168, height=26, corner_radius=6,
            command=self._toggle_report,
        )
        self._report_toggle_btn.grid(row=0, column=3, rowspan=2, sticky="e", padx=(12, 0))

        # ── Body: detection tile + confusion matrix + metrics + timing ─────────
        self._report_body = ctk.CTkFrame(self._report_frame, fg_color=BG_CARD)

        # ── Wajah terdeteksi (face detection) ───────────────────────────────────
        if faces_detected is not None:
            det_tile = ctk.CTkFrame(
                self._report_body, fg_color=C_DETECT_BG, corner_radius=9,
                border_width=1, border_color="#DCE4EA",
            )
            det_tile.pack(fill="x", padx=20, pady=(0, 16))
            ctk.CTkLabel(
                det_tile, text=str(faces_detected),
                font=ctk.CTkFont(FONT_TITLE, 18, "bold"), text_color=C_DETECT_FG,
                fg_color=C_DETECT_BG,
            ).pack(side="left", padx=(16, 12), pady=10)
            ctk.CTkFrame(det_tile, width=1, height=22, fg_color="#DCE4EA").pack(side="left", pady=10)
            ctk.CTkLabel(
                det_tile, text="wajah terdeteksi oleh face detection\n(total cache saat ini)",
                font=ctk.CTkFont(FONT_BODY, 10), text_color=C_SUBTEXT,
                fg_color=C_DETECT_BG, justify="left", anchor="w",
            ).pack(side="left", padx=(12, 16), pady=10)

        # ── Confusion matrix ──────────────────────────────────────────────────
        ctk.CTkLabel(
            self._report_body, text="MATRIKS KONFUSI",
            font=ctk.CTkFont(FONT_BODY, 9, "bold"), text_color=C_MUTED,
            fg_color=BG_CARD, anchor="w",
        ).pack(fill="x", padx=20, pady=(0, 6))

        counts_row = ctk.CTkFrame(self._report_body, fg_color=BG_CARD)
        counts_row.pack(fill="x", padx=20, pady=(0, 16))

        for label, value, bg, fg in [
            ("True Positive",  tp, C_TP_BG, C_TP_FG),
            ("False Positive", fp, C_FP_BG, C_FP_FG),
            ("False Negative", fn, C_FN_BG, C_FN_FG),
            ("True Negative",  tn, C_TN_BG, C_TN_FG),
        ]:
            tile = ctk.CTkFrame(counts_row, fg_color=bg, corner_radius=9)
            tile.pack(side="left", expand=True, fill="x", padx=4)
            ctk.CTkFrame(tile, height=3, fg_color=fg, corner_radius=0).pack(fill="x", side="top")
            ctk.CTkLabel(
                tile, text=str(value),
                font=ctk.CTkFont(FONT_TITLE, 21, "bold"), text_color=fg,
                fg_color=bg,
            ).pack(pady=(9, 0))
            ctk.CTkLabel(
                tile, text=label,
                font=ctk.CTkFont(FONT_BODY, 9, "bold"), text_color=C_SUBTEXT,
                fg_color=bg,
            ).pack(pady=(2, 10))

        # ── Metrics ───────────────────────────────────────────────────────────
        ctk.CTkLabel(
            self._report_body, text="METRIK PERFORMA",
            font=ctk.CTkFont(FONT_BODY, 9, "bold"), text_color=C_MUTED,
            fg_color=BG_CARD, anchor="w",
        ).pack(fill="x", padx=20, pady=(0, 6))

        metrics_row = ctk.CTkFrame(self._report_body, fg_color=BG_CARD)
        metrics_row.pack(fill="x", padx=20, pady=(0, 16))

        metrics = [
            ("Precision", precision, C_SUCCESS if precision >= 0.7 else C_WARNING),
            ("Recall",    recall,    C_SUCCESS if recall    >= 0.7 else C_WARNING),
            ("F1-Score",  f1,        C_SUCCESS if f1        >= 0.7 else C_WARNING),
            ("Accuracy",  accuracy,  C_SUCCESS if accuracy  >= 0.8 else C_WARNING),
        ]

        for name, value, color in metrics:
            tile = ctk.CTkFrame(metrics_row, fg_color=BG_CARD, corner_radius=9,
                                border_width=1, border_color=C_BORDER)
            tile.pack(side="left", expand=True, fill="x", padx=4)
            ctk.CTkLabel(
                tile, text=f"{value:.3f}",
                font=ctk.CTkFont(FONT_TITLE, 17, "bold"), text_color=color,
                fg_color=BG_CARD,
            ).pack(pady=(10, 0))
            ctk.CTkLabel(
                tile, text=name,
                font=ctk.CTkFont(FONT_BODY, 9, "bold"), text_color=C_SUBTEXT,
                fg_color=BG_CARD,
            ).pack(pady=(2, 7))
            prog = ctk.CTkProgressBar(
                tile, height=4, corner_radius=2,
                fg_color="#ECEAE0", progress_color=color,
            )
            prog.set(max(0.0, min(1.0, value)))
            prog.pack(fill="x", padx=12, pady=(0, 10))

        # ── Timing ────────────────────────────────────────────────────────────
        timings = getattr(self, "_index_timings", {})

        def _fmt(secs_f):
            m, s = divmod(secs_f, 60)
            return f"{int(m)}m {s:.1f}s" if m >= 1 else f"{s:.1f}s"

        det_t   = timings.get("detection", 0.0)
        align_t = timings.get("alignment", 0.0)
        emb_t   = timings.get("embedding", 0.0)

        ctk.CTkLabel(
            self._report_body, text="WAKTU PROSES",
            font=ctk.CTkFont(FONT_BODY, 9, "bold"), text_color=C_MUTED,
            fg_color=BG_CARD, anchor="w",
        ).pack(fill="x", padx=20, pady=(0, 6))

        timing_row = ctk.CTkFrame(self._report_body, fg_color=BG_CARD)
        timing_row.pack(fill="x", padx=20, pady=(0, 18))

        timing_tiles = [
            ("Total Indeksasi", _fmt(det_t + align_t + emb_t), C_DETECT_BG, C_DETECT_FG),
            ("Waktu Pencarian", _fmt(self._elapsed),             C_TP_BG, C_TP_FG),
        ]
        if drive_sync_time > 0:
            timing_tiles.append(("Sinkronisasi Drive", _fmt(drive_sync_time), C_FN_BG, C_FN_FG))

        for name, val, bg, fg in timing_tiles:
            tile = ctk.CTkFrame(timing_row, fg_color=bg, corner_radius=9,
                                border_width=1, border_color=C_BORDER)
            tile.pack(side="left", expand=True, fill="x", padx=4)
            ctk.CTkLabel(
                tile, text=val,
                font=ctk.CTkFont(FONT_TITLE, 14, "bold"), text_color=fg,
                fg_color=bg,
            ).pack(pady=(9, 0))
            ctk.CTkLabel(
                tile, text=name,
                font=ctk.CTkFont(FONT_BODY, 9, "bold"), text_color=C_SUBTEXT,
                fg_color=bg,
            ).pack(pady=(2, 9))

        # ── Sumber daya: memory footprint & CPU usage ───────────────────────────
        resource_tiles = []
        if model_init_memory_mb is not None:
            resource_tiles.append((
                "Memory Footprint (Inisialisasi Model)",
                f"{model_init_memory_mb:.1f} MB", C_DETECT_BG, C_DETECT_FG,
            ))
        if memory_incremental_mb is not None:
            resource_tiles.append((
                "Memory Incremental (Proses Indeksasi)",
                f"+{memory_incremental_mb:.1f} MB", C_FN_BG, C_FN_FG,
            ))
        if cpu_usage_percent is not None:
            resource_tiles.append((
                "CPU Usage (Pengindeksan)",
                f"{cpu_usage_percent:.1f}%", C_TP_BG, C_TP_FG,
            ))

        if resource_tiles:
            ctk.CTkLabel(
                self._report_body, text="SUMBER DAYA",
                font=ctk.CTkFont(FONT_BODY, 9, "bold"), text_color=C_MUTED,
                fg_color=BG_CARD, anchor="w",
            ).pack(fill="x", padx=20, pady=(0, 6))

            resource_row = ctk.CTkFrame(self._report_body, fg_color=BG_CARD)
            resource_row.pack(fill="x", padx=20, pady=(0, 18))

            for name, val, bg, fg in resource_tiles:
                tile = ctk.CTkFrame(resource_row, fg_color=bg, corner_radius=9,
                                    border_width=1, border_color=C_BORDER)
                tile.pack(side="left", expand=True, fill="x", padx=4)
                ctk.CTkLabel(
                    tile, text=val,
                    font=ctk.CTkFont(FONT_TITLE, 14, "bold"), text_color=fg,
                    fg_color=bg,
                ).pack(pady=(9, 0))
                ctk.CTkLabel(
                    tile, text=name,
                    font=ctk.CTkFont(FONT_BODY, 9, "bold"), text_color=C_SUBTEXT,
                    fg_color=bg,
                ).pack(pady=(2, 9))

        self._apply_report_collapsed_state()
        self._report_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(8, 0))

    def _toggle_report(self):
        self._report_collapsed = not self._report_collapsed
        self._apply_report_collapsed_state()

    def _apply_report_collapsed_state(self):
        if self._report_collapsed:
            self._report_body.pack_forget()
            self._report_toggle_btn.configure(text="▼ Tampilkan Detail")
        else:
            self._report_body.pack(fill="x", side="top")
            self._report_toggle_btn.configure(text="▲ Sembunyikan Detail")
