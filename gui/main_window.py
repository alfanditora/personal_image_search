import os
import sys
import logging
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# Tambahkan project root ke sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PremiumMainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Desktop Pencarian Foto Pribadi Otomatis - ArcFace")
        self.root.geometry("1100x750")
        self.root.minsize(1000, 680)
        
        # Inisialisasi variabel status
        self.folder_input_path = ""
        self.selfie_path = ""
        self.active_worker = None
        self.detector = None
        self.embedder = None
        self.thumbnails = [] # Menyimpan referensi ImageTk agar tidak dibersihkan garbage collector
        
        # Desain Warna Premium Dark Theme
        self.c_bg = "#121212"          # Latar belakang gelap pekat
        self.c_card = "#1E1E1E"        # Latar belakang card/frame abu-abu gelap
        self.c_input = "#2A2A2A"       # Latar belakang form/tombol non-aktif
        self.c_accent_blue = "#2196F3" # Aksen biru tombol utama
        self.c_accent_green = "#4CAF50"# Aksen hijau tombol sukses
        self.c_text_main = "#FFFFFF"   # Teks putih bersih
        self.c_text_sub = "#AAAAAA"    # Teks abu-abu keterangan
        
        self.root.configure(bg=self.c_bg)
        
        # Konfigurasi TTK Styles agar harmonis dengan tema gelap
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Style Progressbar
        self.style.configure(
            "TProgressbar",
            thickness=12,
            troughcolor=self.c_input,
            background=self.c_accent_blue,
            borderwidth=0
        )
        
        # Style Scrollbar Gelap
        self.style.configure(
            "Vertical.TScrollbar",
            gripcount=0,
            background=self.c_input,
            troughcolor=self.c_bg,
            bordercolor=self.c_bg,
            arrowcolor=self.c_text_main
        )
        
        self.setup_ui_layout()
        
        # Jalankan inisialisasi eager model di background agar UI tidak membeku di awal (freeze-free startup)
        self.disable_all_controls()
        self.update_log("Menyiapkan AI Engine (ArcFace)... Silakan tunggu.")
        threading.Thread(target=self.initialize_ai_engine_eagerly, daemon=True).start()

    def setup_ui_layout(self):
        """Menyusun layout visual grid desktop GUI premium."""
        # 1. Header Frame (Title & Connection Status)
        header_frame = tk.Frame(self.root, bg=self.c_card, height=75, padx=20, pady=10)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="PERSONAL IMAGE SEARCH ENGINE", 
            font=("Helvetica", 16, "bold"), 
            fg=self.c_text_main, 
            bg=self.c_card
        )
        title_label.pack(side=tk.LEFT, anchor=tk.W)
        
        status_label = tk.Label(
            header_frame, 
            text="AI ENGINE: OFFLINE MODE (100% LURING)", 
            font=("Helvetica", 9, "bold"), 
            fg=self.c_accent_green, 
            bg=self.c_card,
            padx=10,
            pady=4,
            relief=tk.FLAT
        )
        status_label.pack(side=tk.RIGHT, anchor=tk.E)
        
        # Container Tengah (Split Panel Kiri & Kanan)
        main_container = tk.Frame(self.root, bg=self.c_bg, padx=15, pady=15)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Grid Column Weight
        main_container.grid_columnconfigure(0, weight=4) # Panel Kontrol & Log
        main_container.grid_columnconfigure(1, weight=5) # Panel Galeri Hasil
        main_container.grid_rowconfigure(0, weight=1)
        
        # ==================== PANEL KIRI (KONTROL & LOG) ====================
        left_panel = tk.Frame(main_container, bg=self.c_bg)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Card 1: Input Jalur Direktori Kumpulan Foto
        dir_card = tk.LabelFrame(left_panel, text=" 1. JALUR DIREKTORI INPUT ", font=("Helvetica", 10, "bold"), fg=self.c_accent_blue, bg=self.c_card, padx=15, pady=15, bd=1, relief=tk.SOLID)
        dir_card.pack(fill=tk.X, pady=(0, 10))
        
        self.btn_select_dir = tk.Button(
            dir_card, 
            text="Pilih Folder Gambar", 
            command=self.select_folder_path,
            font=("Helvetica", 9, "bold"),
            fg=self.c_text_main,
            bg=self.c_input,
            activebackground=self.c_accent_blue,
            activeforeground=self.c_text_main,
            relief=tk.FLAT,
            padx=10,
            pady=6
        )
        self.btn_select_dir.pack(side=tk.LEFT, padx=(0, 10))
        
        self.lbl_dir_path = tk.Label(
            dir_card, 
            text="Folder belum dipilih...", 
            font=("Helvetica", 9, "italic"), 
            fg=self.c_text_sub, 
            bg=self.c_card,
            anchor=tk.W
        )
        self.lbl_dir_path.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.btn_scan = tk.Button(
            dir_card, 
            text="Indeks Foto", 
            command=self.start_indexing_pipeline,
            font=("Helvetica", 9, "bold"),
            fg=self.c_text_main,
            bg=self.c_accent_blue,
            activebackground=self.c_accent_green,
            activeforeground=self.c_text_main,
            relief=tk.FLAT,
            padx=15,
            pady=6
        )
        self.btn_scan.pack(side=tk.RIGHT)
        
        # Card 2: Input Selfie Kueri & Parameter Threshold
        selfie_card = tk.LabelFrame(left_panel, text=" 2. PENCARIAN FOTO SELFIE ", font=("Helvetica", 10, "bold"), fg=self.c_accent_blue, bg=self.c_card, padx=15, pady=15, bd=1, relief=tk.SOLID)
        selfie_card.pack(fill=tk.X, pady=(0, 10))
        
        self.btn_select_selfie = tk.Button(
            selfie_card, 
            text="Pilih Foto Selfie", 
            command=self.select_selfie_query,
            font=("Helvetica", 9, "bold"),
            fg=self.c_text_main,
            bg=self.c_input,
            activebackground=self.c_accent_blue,
            activeforeground=self.c_text_main,
            relief=tk.FLAT,
            padx=10,
            pady=6
        )
        self.btn_select_selfie.pack(side=tk.LEFT, padx=(0, 10))
        
        self.lbl_selfie_path = tk.Label(
            selfie_card, 
            text="Foto selfie belum dipilih...", 
            font=("Helvetica", 9, "italic"), 
            fg=self.c_text_sub, 
            bg=self.c_card,
            anchor=tk.W
        )
        self.lbl_selfie_path.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Slider Threshold (Cosine Distance)
        threshold_frame = tk.Frame(selfie_card, bg=self.c_card)
        threshold_frame.pack(fill=tk.X, pady=(15, 0))
        
        tk.Label(
            threshold_frame, 
            text="Batas Jarak (Threshold):", 
            font=("Helvetica", 9, "bold"), 
            fg=self.c_text_main, 
            bg=self.c_card
        ).pack(side=tk.LEFT)
        
        self.val_threshold = tk.DoubleVar(value=0.40)
        self.sld_threshold = tk.Scale(
            threshold_frame,
            from_=0.10,
            to=1.00,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.val_threshold,
            bg=self.c_card,
            fg=self.c_accent_blue,
            highlightthickness=0,
            activebackground=self.c_accent_blue,
            length=180
        )
        self.sld_threshold.pack(side=tk.LEFT, padx=10)
        
        self.btn_search = tk.Button(
            selfie_card, 
            text="Cari Foto", 
            command=self.start_search_engine,
            font=("Helvetica", 9, "bold"),
            fg=self.c_text_main,
            bg=self.c_accent_green,
            activebackground=self.c_accent_blue,
            activeforeground=self.c_text_main,
            relief=tk.FLAT,
            padx=20,
            pady=6
        )
        self.btn_search.pack(side=tk.RIGHT, pady=(10, 0))

        # Card 3: Indikator Progres & Log Dashboard
        progress_card = tk.LabelFrame(left_panel, text=" STATUS AKTIVITAS & LOG ", font=("Helvetica", 10, "bold"), fg=self.c_accent_blue, bg=self.c_card, padx=15, pady=15, bd=1, relief=tk.SOLID)
        progress_card.pack(fill=tk.BOTH, expand=True)
        
        # Progress Bar & Teks Persentase
        bar_frame = tk.Frame(progress_card, bg=self.c_card)
        bar_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.prog_bar = ttk.Progressbar(bar_frame, mode="determinate", style="TProgressbar")
        self.prog_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.lbl_progress_pct = tk.Label(
            bar_frame, 
            text="0%", 
            font=("Helvetica", 10, "bold"), 
            fg=self.c_text_main, 
            bg=self.c_card,
            width=5
        )
        self.lbl_progress_pct.pack(side=tk.RIGHT)
        
        self.lbl_activity = tk.Label(
            progress_card, 
            text="AI engine siap.", 
            font=("Helvetica", 9, "bold"), 
            fg=self.c_accent_green, 
            bg=self.c_card,
            anchor=tk.W
        )
        self.lbl_activity.pack(fill=tk.X, pady=(0, 10))
        
        # Text Log Console
        log_frame = tk.Frame(progress_card, bg=self.c_card)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.txt_log = tk.Text(
            log_frame, 
            bg=self.c_bg, 
            fg=self.c_text_main, 
            font=("Consolas", 8),
            wrap=tk.WORD,
            bd=0,
            padx=8,
            pady=8
        )
        self.txt_log.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.txt_log.yview)
        log_scroll.pack(fill=tk.Y, side=tk.RIGHT)
        self.txt_log.configure(yscrollcommand=log_scroll.set)
        
        # ==================== PANEL KANAN (GALERI HASIL PENCARIAN) ====================
        right_panel = tk.LabelFrame(main_container, text=" HASIL PENCARIAN FOTO COCOK (KLIK UNTUK MEMBUKA) ", font=("Helvetica", 10, "bold"), fg=self.c_accent_blue, bg=self.c_card, padx=15, pady=15, bd=1, relief=tk.SOLID)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        # Scrollable Canvas untuk Grid Gambar Hasil
        self.canvas_gallery = tk.Canvas(right_panel, bg=self.c_bg, highlightthickness=0)
        self.canvas_gallery.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        gallery_scroll = ttk.Scrollbar(right_panel, orient=tk.VERTICAL, command=self.canvas_gallery.yview)
        gallery_scroll.pack(fill=tk.Y, side=tk.RIGHT)
        self.canvas_gallery.configure(yscrollcommand=gallery_scroll.set)
        
        self.grid_frame = tk.Frame(self.canvas_gallery, bg=self.c_bg)
        self.canvas_gallery.create_window((0, 0), window=self.grid_frame, anchor="nw")
        
        # Bind perubahan ukuran canvas untuk auto-wrap scroll region
        self.grid_frame.bind("<Configure>", lambda e: self.canvas_gallery.configure(scrollregion=self.canvas_gallery.bbox("all")))
        
        # Bind mouse wheel untuk scroll galeri secara halus
        self.canvas_gallery.bind_all("<MouseWheel>", lambda e: self.canvas_gallery.yview_scroll(int(-1 * (e.delta / 120)), "units"))

    # ==================== LOGIKA OPERASI FRONT-END ====================
    def initialize_ai_engine_eagerly(self):
        """Memuat model AI secara eager pada thread latar belakang."""
        try:
            from core.detector import FaceDetector
            from core.embedder import ArcFaceEmbedder
            
            self.detector = FaceDetector()
            self.embedder = ArcFaceEmbedder()
            
            self.root.after(0, self.enable_all_controls)
            self.root.after(0, lambda: self.update_log("AI Engine siap. Status: 100% OFFLINE."))
            self.root.after(0, lambda: self.lbl_activity.configure(text="AI Engine Siap.", fg=self.c_accent_green))
        except Exception as e:
            err_msg = f"Terjadi kegagalan fatal saat inisialisasi AI model: {str(e)}"
            self.root.after(0, lambda: self.update_log(err_msg))
            self.root.after(0, lambda: self.lbl_activity.configure(text="AI Engine ERROR", fg="red"))
            self.root.after(0, lambda: messagebox.showerror("Fatal Error", err_msg))

    def select_folder_path(self):
        """Dialog pasif untuk memilih folder lokal utama kumpulan foto."""
        path = filedialog.askdirectory(title="Pilih Direktori Utama Kumpulan Foto")
        if path:
            self.folder_input_path = os.path.abspath(path)
            # Potong path jika terlalu panjang untuk tampilan UI
            display_path = self.folder_input_path
            if len(display_path) > 40:
                display_path = "..." + display_path[-37:]
            self.lbl_dir_path.configure(text=display_path, font=("Helvetica", 9, "bold"), fg=self.c_text_main)
            self.update_log(f"Jalur direktori terpilih: {self.folder_input_path}")

    def select_selfie_query(self):
        """Dialog pasif untuk memilih 1 file foto kueri selfie acuan."""
        path = filedialog.askopenfilename(
            title="Pilih Berkas Foto Selfie Kueri",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png"), ("All Files", "*.*")]
        )
        if path:
            self.selfie_path = os.path.abspath(path)
            display_path = self.selfie_path
            if len(display_path) > 40:
                display_path = "..." + display_path[-37:]
            self.lbl_selfie_path.configure(text=display_path, font=("Helvetica", 9, "bold"), fg=self.c_text_main)
            self.update_log(f"Kueri foto selfie terpilih: {self.selfie_path}")

    def update_log(self, message: str):
        """Menambahkan pesan teks baris baru ke log console dashboard."""
        self.txt_log.insert(tk.END, message + "\n")
        self.txt_log.see(tk.END)

    def disable_all_controls(self):
        """Menonaktifkan kontrol tombol guna mencegah double-execution."""
        self.btn_select_dir.configure(state=tk.DISABLED, bg=self.c_input)
        self.btn_select_selfie.configure(state=tk.DISABLED, bg=self.c_input)
        self.btn_scan.configure(state=tk.DISABLED, bg=self.c_input)
        self.btn_search.configure(state=tk.DISABLED, bg=self.c_input)
        self.sld_threshold.configure(state=tk.DISABLED)

    def enable_all_controls(self):
        """Mengaktifkan kembali kontrol tombol setelah worker selesai."""
        self.btn_select_dir.configure(state=tk.NORMAL, bg=self.c_input)
        self.btn_select_selfie.configure(state=tk.NORMAL, bg=self.c_input)
        self.btn_scan.configure(state=tk.NORMAL, bg=self.c_accent_blue)
        self.btn_search.configure(state=tk.NORMAL, bg=self.c_accent_green)
        self.sld_threshold.configure(state=tk.NORMAL)

    # ==================== PIPELINE INDEXER BACKGROUND WORKER ====================
    def start_indexing_pipeline(self):
        """Memulai proses background pipeline indexing."""
        if not self.folder_input_path:
            messagebox.showwarning("Peringatan", "Silakan pilih folder gambar terlebih dahulu.")
            return
            
        self.disable_all_controls()
        self.prog_bar["value"] = 0
        self.lbl_progress_pct.configure(text="0%")
        self.lbl_activity.configure(text="Sedang memindai dan membuat indeks...", fg=self.c_accent_blue)
        self.update_log("\n--- Memulai Proses Pindai & Indeks Foto ---")
        
        from gui.workers import PipelineWorker
        self.active_worker = PipelineWorker(
            folder_path=self.folder_input_path,
            on_progress=self.on_indexing_progress,
            on_finished=self.on_indexing_finished,
            on_error=self.on_worker_error,
            detector=self.detector,
            embedder=self.embedder
        )
        self.active_worker.start()

    def on_indexing_progress(self, current: int, total: int, file_name: str):
        """Memperbarui status progress indexing secara thread-safe."""
        pct = int((current / total) * 100)
        self.root.after(0, lambda: self.prog_bar.configure(value=pct))
        self.root.after(0, lambda: self.lbl_progress_pct.configure(text=f"{pct}%"))
        self.root.after(0, lambda: self.lbl_activity.configure(text=f"Memproses {current} dari {total} foto..."))
        self.root.after(0, lambda: self.update_log(f"[{pct}%] Sukses memproses: {file_name}"))

    def on_indexing_finished(self, count: int):
        """Pemberitahuan ketika pipeline indeks selesai."""
        self.active_worker = None
        self.root.after(0, self.enable_all_controls)
        self.root.after(0, lambda: self.prog_bar.configure(value=100))
        self.root.after(0, lambda: self.lbl_progress_pct.configure(text="100%"))
        self.root.after(0, lambda: self.lbl_activity.configure(text="Indeks selesai.", fg=self.c_accent_green))
        self.root.after(0, lambda: self.update_log(f"SUKSES: Selesai mengindeks {count} file foto secara lokal (Zero-DB).\n"))
        self.root.after(0, lambda: messagebox.showinfo("Selesai", f"Proses indeks selesai. Berhasil memproses {count} berkas foto."))

    # ==================== SEARCH ENGINE BACKGROUND WORKER ====================
    def start_search_engine(self):
        """Memulai proses background search engine pencarian wajah."""
        if not self.folder_input_path:
            messagebox.showwarning("Peringatan", "Silakan pilih folder gambar terlebih dahulu.")
            return
        if not self.selfie_path:
            messagebox.showwarning("Peringatan", "Silakan pilih berkas foto selfie kueri terlebih dahulu.")
            return
            
        self.disable_all_controls()
        self.clear_gallery_grid()
        self.prog_bar["value"] = 0
        self.lbl_progress_pct.configure(text="0%")
        self.lbl_activity.configure(text="Menganalisis kemiripan wajah...", fg=self.c_accent_blue)
        self.update_log("\n--- Memulai Pencarian Foto Selfie ---")
        
        from gui.workers import SearchWorker
        threshold = self.val_threshold.get()
        
        self.active_worker = SearchWorker(
            folder_path=self.folder_input_path,
            query_img_path=self.selfie_path,
            threshold=threshold,
            on_progress=self.on_search_progress,
            on_finished=self.on_search_finished,
            on_error=self.on_worker_error,
            detector=self.detector,
            embedder=self.embedder
        )
        self.active_worker.start()

    def on_search_progress(self, status_msg: str):
        """Update realtime status logging search engine."""
        self.root.after(0, lambda: self.update_log(status_msg))
        self.root.after(0, lambda: self.lbl_activity.configure(text=status_msg))

    def on_search_finished(self, matches: list[dict], output_folder: str):
        """Memproses hasil pencocokan setelah search worker selesai."""
        self.active_worker = None
        self.root.after(0, self.enable_all_controls)
        self.root.after(0, lambda: self.prog_bar.configure(value=100))
        self.root.after(0, lambda: self.lbl_progress_pct.configure(text="100%"))
        self.root.after(0, lambda: self.lbl_activity.configure(text=f"Pencarian selesai. Terfilter {len(matches)} foto.", fg=self.c_accent_green))
        
        self.root.after(0, lambda: self.update_log(f"Pencarian selesai. Ditemukan {len(matches)} wajah yang lolos threshold di folder hasil."))
        if output_folder:
            self.root.after(0, lambda: self.update_log(f"Hasil disalin ke: {output_folder}\n"))
            
        # Tampilkan visual hasil di galeri grid
        self.root.after(0, lambda: self.display_matches_in_gallery(matches))

    # ==================== ERROR HANDLING & GALLERY ENGINE ====================
    def on_worker_error(self, err_msg: str):
        """Callback jika background thread melempar eksepsi."""
        self.active_worker = None
        self.root.after(0, self.enable_all_controls)
        self.root.after(0, lambda: self.lbl_activity.configure(text="Proses Gagal", fg="red"))
        self.root.after(0, lambda: self.update_log(f"ERROR: {err_msg}\n"))
        self.root.after(0, lambda: messagebox.showerror("Galat Proses", err_msg))

    def clear_gallery_grid(self):
        """Membersihkan seluruh card gambar di galeri hasil."""
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        self.thumbnails.clear()
        self.canvas_gallery.yview_moveto(0)

    def display_matches_in_gallery(self, matches: list[dict]):
        """Menyusun grid thumbnail premium dari berkas foto hasil pencarian."""
        self.clear_gallery_grid()
        
        if not matches:
            no_result_lbl = tk.Label(
                self.grid_frame,
                text="Tidak ada foto yang cocok ditemukan.",
                font=("Helvetica", 10, "bold"),
                fg=self.c_text_sub,
                bg=self.c_bg,
                pady=50
            )
            no_result_lbl.pack(fill=tk.BOTH, expand=True)
            return
            
        cols = 3 # Tiga kolom thumbnail
        for idx, match in enumerate(matches):
            file_path = match["file_path"]
            file_name = match["file_name"]
            distance = match["cosine_distance"]
            similarity = match["similarity"]
            bbox = match["bbox"]
            
            # Card frame kontainer untuk masing-masing foto
            card = tk.Frame(self.grid_frame, bg=self.c_card, bd=1, relief=tk.SOLID, padx=6, pady=6)
            
            # Grid Position Calculation
            r = idx // cols
            c = idx % cols
            card.grid(row=r, column=c, padx=8, pady=8, sticky="nsew")
            
            # Load and crop thumbnail wajah atau gambar asli secara cepat
            thumb_label = tk.Label(card, bg=self.c_bg, text="Loading...", fg=self.c_text_sub, width=120, height=120)
            thumb_label.pack(fill=tk.BOTH, expand=True)
            
            # Jalankan loading thumbnail di background thread ringan agar rendering grid mulus
            threading.Thread(
                target=self.load_thumbnail_async,
                args=(file_path, bbox, thumb_label),
                daemon=True
            ).start()
            
            # Tambahkan metadata info jarak Cosine di bawah gambar
            info_lbl = tk.Label(
                card, 
                text=f"{file_name}\nSim: {similarity:.2f} | Dist: {distance:.2f}",
                font=("Helvetica", 8, "bold"),
                fg=self.c_accent_green if distance <= 0.35 else self.c_text_main,
                bg=self.c_card,
                pady=4
            )
            info_lbl.pack(fill=tk.X)
            
            # Binding klik agar pengguna dapat langsung membuka foto asli di viewer sistem operasi bawaan (Windows)
            for widget in (thumb_label, info_lbl, card):
                widget.bind("<Button-1>", lambda e, path=file_path: self.open_image_in_os_viewer(path))
                widget.configure(cursor="hand2")

    def load_thumbnail_async(self, file_path: str, bbox: list, label_widget: tk.Label):
        """Memotong wajah berbasis bbox secara async untuk ditampilkan di galeri."""
        try:
            # Muat gambar secara aman
            with Image.open(file_path) as img:
                # Potong bagian wajah menggunakan bbox koordinat asli (opsional/lebih premium jika tersedia)
                x, y, w, h = bbox
                if w > 0 and h > 0:
                    # Tambahkan margin padding 20% agar potongan kepala terlihat alami di grid
                    pad_x = int(w * 0.15)
                    pad_y = int(h * 0.15)
                    
                    left = max(0, x - pad_x)
                    top = max(0, y - pad_y)
                    right = min(img.width, x + w + pad_x)
                    bottom = min(img.height, y + h + pad_y)
                    
                    face_crop = img.crop((left, top, right, bottom))
                else:
                    face_crop = img
                    
                # Ubah ukuran menjadi thumbnail 120x120 yang seragam
                face_crop.thumbnail((120, 120), Image.Resampling.LANCZOS)
                
                # Buat photo image
                tk_thumb = ImageTk.PhotoImage(face_crop)
                
                # Simpan referensi objek
                self.thumbnails.append(tk_thumb)
                
                # Update widget label secara thread-safe menggunakan UI thread
                self.root.after(0, lambda: label_widget.configure(image=tk_thumb, text=""))
        except Exception as e:
            logger.warning(f"Gagal memuat thumbnail async untuk {os.path.basename(file_path)}: {str(e)}")
            self.root.after(0, lambda: label_widget.configure(text="[No preview]"))

    def open_image_in_os_viewer(self, file_path: str):
        """Membuka foto asli secara instan menggunakan aplikasi default OS (Windows)."""
        try:
            if os.path.exists(file_path):
                logger.info(f"Membuka berkas citra asli: {file_path}")
                os.startfile(file_path)
            else:
                messagebox.showerror("Galat", "Berkas asli tidak ditemukan.")
        except Exception as e:
            logger.error(f"Gagal membuka berkas citra di viewer bawaan: {str(e)}")
            messagebox.showerror("Galat", f"Gagal membuka gambar: {str(e)}")

    def on_safe_close(self):
        """Membersihkan thread background secara aman sebelum aplikasi ditutup (anti-leak)."""
        if self.active_worker is not None:
            if messagebox.askyesno("Tutup Aplikasi", "Proses pemrosesan AI sedang berjalan. Apakah Anda yakin ingin membatalkan dan keluar?"):
                self.active_worker.stop()
                logger.info("Menghentikan background thread secara paksa sebelum keluar.")
                self.root.destroy()
        else:
            self.root.destroy()
