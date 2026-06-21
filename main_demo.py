import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import customtkinter as ctk
from gui.demo_window import DemoApp

if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    app = DemoApp()
    app.mainloop()
