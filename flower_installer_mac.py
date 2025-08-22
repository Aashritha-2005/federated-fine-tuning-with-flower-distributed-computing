#!/usr/bin/env python3
"""
Flower GUI Installer - One-click Flower framework installation
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import threading
import sys
import platform
import os

class FlowerInstaller:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Flower Framework Installer")
        self.root.geometry("600x500")
        self.root.configure(bg='#d9d9d9')  
        
        self.installing = False
        self.setup_ui()
        
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg='#6366f1', height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="🌸 Flower Framework Installer", 
                               font=('Arial', 18, 'bold'), fg='white', bg='#6366f1')
        title_label.pack(pady=20)
        
        # Main Content
        main_frame = tk.Frame(self.root, bg='#d9d9d9')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        info_text = f"Platform: {platform.system()} {platform.machine()}\nPython: {sys.version.split()[0]}"
        info_label = tk.Label(main_frame, text=info_text, font=('Arial', 10, 'bold'), 
                              bg='#d9d9d9', fg='black')
        info_label.pack(anchor='w', pady=(0, 10))
        
        desc_text = ("Welcome to the Flower Framework installer!\n"
                     "This tool will install Flower and all required dependencies.\n"
                     "Perfect for federated learning projects and research.")
        desc_label = tk.Label(main_frame, text=desc_text, font=('Arial', 11, 'bold'), 
                              bg='#d9d9d9', fg='black', wraplength=550, justify='left')
        desc_label.pack(anchor='w', pady=(0, 20))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill='x', pady=(0, 10))
        
        # Install button
        self.install_btn = tk.Button(main_frame, text="🚀 Install Flower Framework", 
                                     font=('Arial', 12, 'bold'), bg='#10b981', fg='black',
                                     command=self.start_installation, height=2, cursor='hand2')
        self.install_btn.pack(fill='x', pady=(0, 20))
        
        # Log label
        log_label = tk.Label(main_frame, text="Installation Log:", font=('Arial', 10, 'bold'), 
                             bg='#d9d9d9', fg='black', anchor='w')
        log_label.pack(anchor='w')
        
        # Log box
        self.log_text = scrolledtext.ScrolledText(main_frame, height=15, font=('Consolas', 9),
                                                  bg='#1f2937', fg='#10b981', insertbackground='white')
        self.log_text.pack(fill='both', expand=True)
        self.log_text.insert('end', "Ready to install Flower framework...\n")
        
    def log_message(self, message):
        """Add message to log with timestamp"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert('end', f"[{timestamp}] {message}\n")
        self.log_text.see('end')
        self.root.update()
        
    def run_command(self, command, description):
        """Execute command and log output"""
        self.log_message(f"Starting: {description}")
        self.log_message(f"Command: {command}")
        
        try:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, 
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
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
        """Main installation process"""
        self.installing = True
        self.install_btn.config(state='disabled', text="Installing...", bg='#6b7280')
        self.progress.start()
        
        try:
            # Check Python
            self.log_message("🔍 Checking Python installation...")
            if not self.run_command("python --version", "Python version check"):
                if not self.run_command("python3 --version", "Python3 version check"):
                    raise Exception("Python not found! Please install Python first.")
            
            # Upgrade pip
            self.log_message("📦 Upgrading pip...")
            pip_cmd = "python -m pip install --upgrade pip" if platform.system() == "Windows" else "python3 -m pip install --upgrade pip"
            self.run_command(pip_cmd, "Pip upgrade")
            
            # Install Flower
            self.log_message("🌸 Installing Flower framework...")
            install_cmd = "python -m pip install flwr" if platform.system() == "Windows" else "python3 -m pip install flwr"
            if not self.run_command(install_cmd, "Flower installation"):
                raise Exception("Failed to install Flower")
            
            # Install common ML dependencies
            self.log_message("🧠 Installing ML dependencies...")
            deps_cmd = "python -m pip install torch torchvision numpy matplotlib" if platform.system() == "Windows" else "python3 -m pip install torch torchvision numpy matplotlib"
            self.run_command(deps_cmd, "ML dependencies installation")
            
            # Verify installation
            self.log_message("✅ Verifying installation...")

            if platform.system() == "Windows":
                verify_cmd = "python -c \"import flwr; print(f'Flower {flwr.__version__} installed successfully!')\""
            else:
                verify_cmd = "python3 -c \"import flwr; print(f'Flower {flwr.__version__} installed successfully!')\""

            if self.run_command(verify_cmd, "Installation verification"):
                self.log_message("🎉 INSTALLATION COMPLETED SUCCESSFULLY!")
                self.log_message("You can now start building Flower federated learning applications!")
                messagebox.showinfo("Success", "Flower framework installed successfully!\nCheck the log for details.")
            else:
                self.log_message("⚠ Installation completed but verification failed")

                
        except Exception as e:
            self.log_message(f"❌ Installation failed: {str(e)}")
            messagebox.showerror("Error", f"Installation failed: {str(e)}")
        
        finally:
            self.progress.stop()
            self.install_btn.config(state='normal', text="🚀 Install Flower Framework", bg='#10b981')
            self.installing = False
    
    def start_installation(self):
        """Start installation in separate thread"""
        if not self.installing:
            thread = threading.Thread(target=self.install_flower)
            thread.daemon = True
            thread.start()
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = FlowerInstaller()
    app.run()