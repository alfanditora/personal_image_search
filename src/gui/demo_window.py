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
from tkinter import filedialog, messagebox
import customtkinter as ctk

from gui.main_window import (
    MainApp,
    BG_MAIN, BG_SIDEBAR, BG_CARD, BG_CARD_HV, BG_INPUT,
    C_ACCENT, C_ACCENT_HV, C_SUCCESS, C_SUCCESS_HV,
    C_ERROR, C_WARNING, C_TEXT, C_SUBTEXT, C_MUTED,
    C_BORDER, C_HDR_BG,
    THUMB_W, THUMB_H,
)

logger = logging.getLogger(__name__)

C_REPORT_BG = "#F0F9FF"
C_TP_BG     = "#DCFCE7"
C_FP_BG     = "#FEE2E2"
C_FN_BG     = "#FEF9C3"
C_TN_BG     = "#F1F5F9"

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
        self._query_person      = None   # StringVar — created in _build_sidebar after Tk root
        super().__init__()
        self.title("Personal Image Search — Demo")

    # ── Override header ───────────────────────────────────────────────────────

    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=C_HDR_BG, height=60, corner_radius=0)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        title_frame = ctk.CTkFrame(hdr, fg_color="transparent")
        title_frame.pack(side="left", padx=24, pady=10)

        ctk.CTkLabel(
            title_frame, text="Personal Image Search",
            font=ctk.CTkFont("Segoe UI", 17, "bold"),
            text_color="#0F172A",
        ).pack(side="left")

        ctk.CTkLabel(
            title_frame, text="  DEMO",
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
            text_color="#1E40AF",
            fg_color="#DBEAFE",
            corner_radius=4,
        ).pack(side="left", padx=(8, 0))

        self.engine_badge = ctk.CTkLabel(
            hdr, text="● SCRFD + ArcFace  |  OFFLINE",
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
            text_color="#15803D",
        )
        self.engine_badge.pack(side="right", padx=24)

    # ── Override sidebar — Ground Truth + Query Person + Threshold slider ─────

    def _build_sidebar(self, parent):
        # DoubleVar and StringVar must be created after the Tk root exists
        self._threshold    = ctk.DoubleVar(value=0.60)
        self._query_person = ctk.StringVar(value="A")

        wrapper = ctk.CTkFrame(parent, width=296, fg_color=C_BORDER, corner_radius=0)
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

        self.folder_label = ctk.CTkLabel(
            sb, text="Belum dipilih",
            font=ctk.CTkFont("Segoe UI", 11), text_color=C_MUTED,
            anchor="w", wraplength=252,
        )
        self.folder_label.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 6)); r += 1

        ctk.CTkButton(
            sb, text="Pilih Folder",
            font=ctk.CTkFont("Segoe UI", 12),
            fg_color=BG_INPUT, hover_color=C_BORDER, text_color=C_TEXT,
            height=36, corner_radius=8, border_width=1, border_color=C_BORDER,
            command=self._select_folder,
        ).grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 6)); r += 1

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
            anchor="w", wraplength=252,
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

        # ── Ground Truth ──────────────────────────────────────────────────────
        self._section_label(sb, "GROUND TRUTH").grid(
            row=r, column=0, sticky="w", padx=16, pady=(0, 4)); r += 1

        self._gt_status_label = ctk.CTkLabel(
            sb, text="Belum dimuat",
            font=ctk.CTkFont("Segoe UI", 11), text_color=C_MUTED,
            anchor="w", wraplength=252, fg_color=BG_SIDEBAR,
        )
        self._gt_status_label.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 6)); r += 1

        ctk.CTkButton(
            sb, text="Pilih File Ground Truth",
            font=ctk.CTkFont("Segoe UI", 12),
            fg_color=BG_INPUT, hover_color=C_BORDER, text_color=C_TEXT,
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
            fg_color=BG_INPUT,
            selected_color=C_ACCENT,
            selected_hover_color=C_ACCENT_HV,
            unselected_color=BG_INPUT,
            unselected_hover_color=C_BORDER,
            text_color=C_TEXT,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
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
            font=ctk.CTkFont("Segoe UI", 12, "bold"), text_color=C_ACCENT,
            fg_color=BG_SIDEBAR,
        )
        self.thresh_val_label.grid(row=0, column=1, sticky="e")

        ctk.CTkSlider(
            sb, from_=0.10, to=1.00, number_of_steps=18,
            variable=self._threshold,
            fg_color=C_BORDER,
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
            font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color=C_ACCENT,
            fg_color=BG_SIDEBAR,
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
            anchor="w", wraplength=252, fg_color=BG_SIDEBAR,
        )
        self.status_label.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 4)); r += 1

        self.time_label = ctk.CTkLabel(
            sb, text="",
            font=ctk.CTkFont("Segoe UI", 10), text_color=C_MUTED,
            anchor="w", fg_color=BG_SIDEBAR,
        )
        self.time_label.grid(row=r, column=0, sticky="ew", padx=16, pady=(0, 24))

    # ── Override gallery to include report panel row ──────────────────────────

    def _build_gallery(self, parent):
        self._gal_frame = ctk.CTkFrame(parent, fg_color=BG_MAIN, corner_radius=0)
        self._gal_frame.grid(row=0, column=1, sticky="nsew")
        self._gal_frame.grid_columnconfigure(0, weight=1)
        self._gal_frame.grid_rowconfigure(2, weight=1)

        hdr_row = ctk.CTkFrame(self._gal_frame, fg_color=BG_MAIN, height=52)
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

        # Report panel (row=1, hidden until metrics are ready)
        self._report_frame = ctk.CTkFrame(
            self._gal_frame, fg_color=C_REPORT_BG, corner_radius=10,
            border_width=1, border_color=C_BORDER,
        )

        self.gallery_scroll = ctk.CTkScrollableFrame(
            self._gal_frame, fg_color=BG_MAIN,
            scrollbar_button_color=C_BORDER,
            scrollbar_button_hover_color=C_MUTED,
            scrollbar_fg_color=BG_INPUT,
        )
        self.gallery_scroll.grid(row=2, column=0, sticky="nsew", padx=(20, 8), pady=(8, 16))

        self._show_empty_state()

    # ── Threshold callback ────────────────────────────────────────────────────

    def _on_threshold_change(self, value):
        rounded = round(value / 0.05) * 0.05
        self.thresh_val_label.configure(text=f"{rounded:.2f}")

    # ── Override search to read slider threshold ──────────────────────────────

    def _start_search(self):
        if not self.folder_path:
            messagebox.showwarning("Folder Belum Dipilih", "Pilih folder foto terlebih dahulu.")
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
        self._hide_report()
        self._start_timer()

        threshold = round(self._threshold.get() / 0.05) * 0.05

        from gui.workers import SearchWorker
        self.active_worker = SearchWorker(
            folder_path=self.folder_path,
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

            if not matches:
                self._set_status("Tidak ada foto yang cocok.", C_WARNING)
                self._show_empty_state(no_results=True)
                self.result_count_label.configure(text="0 hasil")
                self._hide_report()
                return

            self._set_status(f"Ditemukan {len(matches)} foto cocok.", C_SUCCESS)
            self.result_count_label.configure(text=f"{len(matches)} foto")

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
                self._compute_and_show_report(matches, gt_basenames)

            elif self._ground_truth:
                # Flat GT
                gt_basenames = {k for k, v in self._ground_truth.items() if v}
                self._compute_and_show_report(matches, gt_basenames)

            else:
                self._hide_report()

            self._display_gallery(matches, gt_basenames=gt_basenames)
        self.after(0, _u)

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

        self._render_report(tp, fp, fn, tn, precision, recall, f1, accuracy,
                            person_label=person_label,
                            total_images=total_images,
                            positive_count=len(gt_positive))

    def _render_report(self, tp, fp, fn, tn, precision, recall, f1, accuracy,
                       person_label=None, total_images=None, positive_count=None):
        for w in self._report_frame.winfo_children():
            w.destroy()

        # ── Header ────────────────────────────────────────────────────────────
        hdr_row = ctk.CTkFrame(self._report_frame, fg_color=C_REPORT_BG)
        hdr_row.pack(fill="x", padx=16, pady=(12, 8))
        hdr_row.grid_columnconfigure(0, weight=1)

        title = "LAPORAN EVALUASI"
        if person_label:
            title += f"  —  {person_label}"
        ctk.CTkLabel(
            hdr_row, text=title,
            font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color=C_SUBTEXT,
            fg_color=C_REPORT_BG, anchor="w",
        ).grid(row=0, column=0, sticky="w")

        if total_images is not None and positive_count is not None:
            ctk.CTkLabel(
                hdr_row,
                text=f"{positive_count} positif dari {total_images} gambar",
                font=ctk.CTkFont("Segoe UI", 9), text_color=C_MUTED,
                fg_color=C_REPORT_BG,
            ).grid(row=0, column=1, sticky="e")

        # ── Confusion counts ──────────────────────────────────────────────────
        counts_row = ctk.CTkFrame(self._report_frame, fg_color=C_REPORT_BG)
        counts_row.pack(fill="x", padx=16, pady=(0, 8))

        for label, value, bg in [
            ("True Positive",  tp, C_TP_BG),
            ("False Positive", fp, C_FP_BG),
            ("False Negative", fn, C_FN_BG),
            ("True Negative",  tn, C_TN_BG),
        ]:
            tile = ctk.CTkFrame(counts_row, fg_color=bg, corner_radius=8)
            tile.pack(side="left", expand=True, fill="x", padx=4)
            ctk.CTkLabel(
                tile, text=str(value),
                font=ctk.CTkFont("Segoe UI", 20, "bold"), text_color=C_TEXT,
                fg_color=bg,
            ).pack(pady=(8, 0))
            ctk.CTkLabel(
                tile, text=label,
                font=ctk.CTkFont("Segoe UI", 9), text_color=C_SUBTEXT,
                fg_color=bg,
            ).pack(pady=(0, 8))

        # ── Metrics ───────────────────────────────────────────────────────────
        metrics_row = ctk.CTkFrame(self._report_frame, fg_color=C_REPORT_BG)
        metrics_row.pack(fill="x", padx=16, pady=(0, 8))

        mins, secs = divmod(self._elapsed, 60)
        time_str = f"{int(mins)}m {secs:.1f}s" if mins >= 1 else f"{self._elapsed:.1f}s"

        metrics = [
            ("Precision",    f"{precision:.3f}", C_SUCCESS if precision >= 0.7 else C_WARNING),
            ("Recall",       f"{recall:.3f}",    C_SUCCESS if recall    >= 0.7 else C_WARNING),
            ("F1-Score",     f"{f1:.3f}",        C_SUCCESS if f1        >= 0.7 else C_WARNING),
            ("Accuracy",     f"{accuracy:.3f}",  C_SUCCESS if accuracy  >= 0.8 else C_WARNING),
            ("Waktu Cari",   time_str,            C_ACCENT),
        ]

        for name, val, color in metrics:
            tile = ctk.CTkFrame(metrics_row, fg_color=BG_CARD, corner_radius=8,
                                border_width=1, border_color=C_BORDER)
            tile.pack(side="left", expand=True, fill="x", padx=4)
            ctk.CTkLabel(
                tile, text=val,
                font=ctk.CTkFont("Segoe UI", 16, "bold"), text_color=color,
                fg_color=BG_CARD,
            ).pack(pady=(8, 0))
            ctk.CTkLabel(
                tile, text=name,
                font=ctk.CTkFont("Segoe UI", 9), text_color=C_SUBTEXT,
                fg_color=BG_CARD,
            ).pack(pady=(0, 8))

        # ── Indexing timing breakdown ─────────────────────────────────────────
        timings = getattr(self, "_index_timings", {})

        timing_hdr = ctk.CTkFrame(self._report_frame, fg_color=C_REPORT_BG)
        timing_hdr.pack(fill="x", padx=16, pady=(4, 4))
        timing_hdr.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            timing_hdr, text="TIMING INDEXING",
            font=ctk.CTkFont("Segoe UI", 10, "bold"), text_color=C_SUBTEXT,
            fg_color=C_REPORT_BG, anchor="w",
        ).grid(row=0, column=0, sticky="w")

        if timings and timings.get("new_images", 0) > 0:
            n = timings["new_images"]
            ctk.CTkLabel(
                timing_hdr,
                text=f"{n} gambar baru diproses",
                font=ctk.CTkFont("Segoe UI", 9), text_color=C_MUTED,
                fg_color=C_REPORT_BG,
            ).grid(row=0, column=1, sticky="e")
        else:
            ctk.CTkLabel(
                timing_hdr,
                text="semua dari cache  ·  tidak ada indexing baru",
                font=ctk.CTkFont("Segoe UI", 9), text_color=C_MUTED,
                fg_color=C_REPORT_BG,
            ).grid(row=0, column=1, sticky="e")

        timing_row = ctk.CTkFrame(self._report_frame, fg_color=C_REPORT_BG)
        timing_row.pack(fill="x", padx=16, pady=(0, 12))

        def _fmt(secs_f):
            m, s = divmod(secs_f, 60)
            return f"{int(m)}m {s:.1f}s" if m >= 1 else f"{s:.1f}s"

        det_t   = timings.get("detection", 0.0)
        align_t = timings.get("alignment", 0.0)
        emb_t   = timings.get("embedding", 0.0)
        total_t = det_t + align_t + emb_t

        timing_tiles = [
            ("Deteksi Wajah",   _fmt(det_t),   "#EFF6FF", "#1D4ED8"),
            ("Alignment",       _fmt(align_t),  "#F0FDF4", "#15803D"),
            ("Ekstraksi Fitur", _fmt(emb_t),    "#FFF7ED", "#C2410C"),
            ("Total Pipeline",  _fmt(total_t),  "#F5F3FF", "#6D28D9"),
        ]

        for name, val, bg, fg in timing_tiles:
            tile = ctk.CTkFrame(timing_row, fg_color=bg, corner_radius=8,
                                border_width=1, border_color=C_BORDER)
            tile.pack(side="left", expand=True, fill="x", padx=4)
            ctk.CTkLabel(
                tile, text=val,
                font=ctk.CTkFont("Segoe UI", 14, "bold"), text_color=fg,
                fg_color=bg,
            ).pack(pady=(8, 0))
            ctk.CTkLabel(
                tile, text=name,
                font=ctk.CTkFont("Segoe UI", 9), text_color=C_SUBTEXT,
                fg_color=bg,
            ).pack(pady=(0, 8))

        self._report_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(8, 0))
