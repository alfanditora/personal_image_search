import tkinter as tk
from gui.main_window import PremiumMainWindow

if __name__ == "__main__":
    # Inisialisasi jendela utama Tkinter
    root = tk.Tk()
    app = PremiumMainWindow(root)
    
    # Hubungkan tombol silang pojok window dengan shutdown sequence yang aman
    root.protocol("WM_DELETE_WINDOW", app.on_safe_close)
    root.mainloop()
