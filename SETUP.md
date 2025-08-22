# 🌸 Flower GUI Installer - Setup Guide

One-click installer to set up the Flower Federated Learning Framework with required ML dependencies — designed for students, colleges, and research teams.

## 📁 Project Files

Make sure you have the following in your folder:

```
flower_gui_installer/
├── flower_installer.py      ← The main GUI installer script
├── SETUP.md                 ← (this file)
├── README.md                ← (optional usage info)
```

## 🧩 Requirements

✅ **Python must be installed first.**

| OS      | Python Version | Required Modules | Tkinter                                      |
|---------|----------------|------------------|----------------------------------------------|
| Windows | 3.7+           | tkinter, pip     | Installed by default (with official installer) |
| macOS   | 3.7+           | tkinter, pip     | Included by default                          |
| Linux   | 3.7+           | tkinter, pip     | Run `sudo apt install python3-tk` if missing |

## 🛠️ 1. Install Python (if not already installed)

### Windows / macOS:
- Download from: https://www.python.org/downloads/
- ✅ During install, check **"Add Python to PATH"**
- For GUI to work, ensure **"tcl/tk and IDLE"** is checked

### Linux:
```bash
sudo apt update
sudo apt install python3 python3-tk python3-pip
```

## 🔍 2. Verify Python & Tkinter Installation

Open a terminal / CMD:

```bash
python -m tkinter
```

✅ **If a blank GUI window appears, tkinter is working.**

If not:
- Reinstall Python
- On Linux: `sudo apt install python3-tk`

## 🚀 3. Run the Installer

**⚠️ Do not double-click the file in file explorer!** Always run it via terminal to see errors.

### Windows:
```cmd
cd path\to\flower_gui_installer
python flower_installer.py
```

### macOS/Linux:
```bash
cd path/to/flower_gui_installer
python3 flower_installer.py
```

## 📦 What It Installs

The GUI will install:
- **flwr** → The Flower federated learning framework
- **torch, torchvision, numpy, matplotlib** → Common ML dependencies
- **pip will be upgraded**

## 📋 Installation Log

You will see real-time logs in the app window for:
- Checking Python
- Installing dependencies
- Success or failure

At the end, a success message will confirm installation.

## 💡 Post-Installation

To use Flower:

1. **Start a server:**
   ```bash
   python3 -m flwr
   ```
   (or write your own server.py)(Refer README.md)

2. **Create a client.py** that connects to the server

3. **Refer to official docs:** https://flower.dev/docs/

## ❗ Troubleshooting

| Issue | Fix |
|-------|-----|
| GUI doesn't open | Run via terminal; check Python/tkinter install |
| Log says "Python not found" | Ensure `python` or `python3` is in PATH |
| Flower installed but not detected | Check correct Python version was used |
| tkinter not found (Linux) | Run: `sudo apt install python3-tk` |
| GUI closes instantly (Windows) | Run via terminal, not by double-click |

### 🔧 Installation Stops Mid-Process

**If installation freezes or stops:**

1. **Close the installer window**
2. **Check your internet connection**
3. **Try running manually:**

**Windows:**
```cmd
python -m pip install --upgrade pip
python -m pip install flwr
python -m pip install torch torchvision numpy matplotlib
```

**macOS/Linux:**
```bash
python3 -m pip install --upgrade pip
python3 -m pip install flwr
python3 -m pip install torch torchvision numpy matplotlib
```

### 🚫 Permission Errors

**Windows:**
- Run Command Prompt as Administrator
- Or add `--user` flag: `python -m pip install --user flwr`

**macOS/Linux:**
- Use `sudo` if needed: `sudo python3 flower_installer.py`
- Or add `--user` flag: `python3 -m pip install --user flwr`

### 🌐 Network Issues

**If downloads fail:**
- Check internet connection
- Try different network (mobile hotspot)
- Wait and retry later (PyPI servers might be busy)

### 🐍 Python Not Found

**Windows:**
1. Download Python from python.org
2. During install, check "Add Python to PATH"
3. Restart Command Prompt
4. Try `py flower_installer.py` instead

**macOS:**
1. Install Python from python.org
2. Or use Homebrew: `brew install python3`
3. Try `python flower_installer.py` instead of `python3`

**Linux:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-tk
```

## 📁 Optional: Make It Clickable for Windows Users

Create a file named `run_installer.bat`:

```bat
@echo off
python flower_installer.py
pause
```

Double-click this `.bat` file to run the installer.

## 🧪 Want a One-Click .exe?

You can convert this GUI installer to a `.exe` file using pyinstaller:

```bash
pip install pyinstaller
pyinstaller --noconsole --onefile flower_installer.py
```

The `.exe` will be in `dist/flower_installer.exe`

## ✅ Verify Installation

After installation, test if Flower works:

```python
# Save as test.py and run it
import flwr as fl
print(f"Flower version: {fl.__version__}")
print("✅ Flower is ready!")
```

## 🆘 Still Having Problems?

**Before asking for help:**

1. **Run via terminal** (not double-click)
2. **Take a screenshot** of any error messages
3. **Check Python version:** `python --version`
4. **Share the exact error** message you see

**Common student mistakes:**
- Double-clicking the `.py` file instead of running via terminal
- Not having Python installed or in PATH
- Running without internet connection
- Not having administrator privileges when needed

## 🎓 For Students

**After successful installation:**
- [ ] Test with the verification script above
- [ ] Check out Flower tutorials: https://flower.dev/docs/tutorials/
- [ ] Join the community: https://flower.dev/join-slack/
