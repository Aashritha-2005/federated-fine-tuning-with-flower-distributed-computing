#!/usr/bin/env python3
"""
Modern Flower GUI Installer 
"""
# Make sure to install ttkbootstrap to use this code
    """ pip install ttkbootstrap """

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledText

import subprocess
import threading
import sys
import platform
import datetime
from tkinter import messagebox


class FlowerInstaller:
    def __init__(self):
        self.root = ttk.Window(themename="darkly")  
        self.root.title("🌸 Flower Framework Installer")
        self.root.geometry("700x600")
        self.installing = False
        self.setup_ui()

    def setup_ui(self):
        header = ttk.Frame(self.root, style="CustomHeader.TFrame")
        header.pack(fill="x", pady=(0, 10))
        header_label = ttk.Label(
            header,
            text="🌸 Flower Framework Installer",
            font=("Segoe UI", 20, "bold"),
            anchor="center",
            background="#a618e3",  
            foreground="white"
        )
        header_label.pack(fill="x", pady=10)

        # Info
        info = f"Platform: {platform.system()} {platform.machine()}\nPython: {sys.version.split()[0]}"
        self.info_label = ttk.Label(self.root, text=info, font=("Segoe UI", 10, "bold"))
        self.info_label.pack(pady=(0, 10), padx=10, anchor="w")

        desc = (
            "Welcome to the Flower Framework installer!\n"
            "This tool will install Flower and all required dependencies.\n"
            "Perfect for federated learning projects and research."
        )
        self.desc_label = ttk.Label(self.root, text=desc, font=("Segoe UI", 11), wraplength=650, justify="left")
        self.desc_label.pack(pady=(0, 15), padx=10, anchor="w")

        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='indeterminate', bootstyle="success-striped")
        self.progress.pack(fill="x", padx=10, pady=10)

        # Install Button
        self.install_btn = ttk.Button(
        self.root,
        text="🚀 Install Flower Framework",
        bootstyle="info",  # or "secondary", "light"
        command=self.start_installation
)

        self.install_btn.pack(fill="x", padx=10, pady=(0, 20), ipady=5)

        # Log Label
        self.log_label = ttk.Label(self.root, text="📜 Installation Log:", font=("Segoe UI", 10, "bold"))
        self.log_label.pack(anchor="w", padx=10)

        # Log Box
        self.log_text = ScrolledText(self.root, height=18, autohide=True)
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        self.log_text.insert("end", "Ready to install Flower framework...\n")

        # Style override for header
        style = ttk.Style()
        style.configure("CustomHeader.TFrame", background="#4ea8de")

    def log_message(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")
        self.root.update()

    def run_command(self, command, description):
        self.log_message(f"Starting: {description}")
        self.log_message(f"Command: {command}")

        try:
            process = subprocess.Popen(command, shell=True,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            for line in process.stdout:
                self.log_message(line.strip())

            process.wait()
            if process.returncode == 0:
                self.log_message(f"✅ {description} completed successfully!")
                return True
            else:
                self.log_message(f"❌ {description} failed with code {process.returncode}")
                return False

        except Exception as e:
            self.log_message(f"❌ Error: {str(e)}")
            return False

    def install_flower(self):
        self.installing = True
        self.install_btn.config(state="disabled", text="Installing...")
        self.progress.start()

        try:
            # Python check
            self.log_message("🔍 Checking Python installation...")
            if not self.run_command("python --version", "Python version check"):
                if not self.run_command("python3 --version", "Python3 version check"):
                    raise Exception("Python not found! Please install Python first.")

            # Pip upgrade
            self.log_message("📦 Upgrading pip...")
            pip_cmd = "python -m pip install --upgrade pip" if platform.system() == "Windows" else "python3 -m pip install --upgrade pip"
            self.run_command(pip_cmd, "Pip upgrade")

            # Install Flower
            self.log_message("🌸 Installing Flower framework...")
            install_cmd = "python -m pip install flwr" if platform.system() == "Windows" else "python3 -m pip install flwr"
            if not self.run_command(install_cmd, "Flower installation"):
                raise Exception("Failed to install Flower")

            # ML Dependencies
            self.log_message("🧠 Installing ML dependencies...")
            deps_cmd = "python -m pip install torch torchvision numpy matplotlib" if platform.system() == "Windows" else "python3 -m pip install torch torchvision numpy matplotlib"
            self.run_command(deps_cmd, "ML dependencies installation")

            # Verify
            self.log_message("✅ Verifying installation...")
            verify_cmd = (
                "python -c \"import flwr; print(f'Flower {flwr.__version__} installed successfully!')\""
                if platform.system() == "Windows"
                else "python3 -c \"import flwr; print(f'Flower {flwr.__version__} installed successfully!')\""
            )

            if self.run_command(verify_cmd, "Installation verification"):
                self.log_message("🎉 INSTALLATION COMPLETED SUCCESSFULLY!")
                self.log_message("You can now start building federated learning applications!")
                messagebox.showinfo("Success", "Flower framework installed successfully!\nCheck the log for details.")
            else:
                self.log_message("⚠ Installation completed but verification failed")

        except Exception as e:
            self.log_message(f"❌ Installation failed: {str(e)}")
            messagebox.showerror("Error", f"Installation failed:\n{str(e)}")

        finally:
            self.progress.stop()
            self.install_btn.config(state="normal", text="🚀 Install Flower Framework")
            self.installing = False

    def start_installation(self):
        if not self.installing:
            thread = threading.Thread(target=self.install_flower)
            thread.daemon = True
            thread.start()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = FlowerInstaller()
    app.run()
