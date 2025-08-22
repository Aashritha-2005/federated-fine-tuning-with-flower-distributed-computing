# 🌸 Flower Federated Learning with ALBERT

A federated learning implementation using Flower framework and ALBERT model for sentiment analysis on IMDB dataset.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project demonstrates federated learning using:
- **Flower Framework**: For federated learning orchestration
- **ALBERT Model**: Pre-trained transformer for sequence classification
- **IMDB Dataset**: Movie reviews sentiment analysis (2% subset for demo)
- **PyTorch**: Deep learning framework

## ✨ Features

- 🤖 **ALBERT Model**: Uses pre-trained ALBERT-base-v2 for sentiment classification
- 🔄 **Federated Learning**: Distributed training across multiple clients
- 📊 **Real-time Logging**: Detailed progress tracking and metrics
- 🌐 **Network Support**: Works across different machines on same network
- 🎛️ **Configurable**: Easy to modify training parameters
- 📈 **Metrics Tracking**: Accuracy and loss monitoring per round

## 🛠️ Requirements

### System Requirements
- Python 3.8 or higher
- 4GB+ RAM recommended
- Internet connection for initial model download

### Python Dependencies
```
torch>=1.9.0
transformers>=4.20.0
flwr==1.6.0
datasets>=2.0.0
scikit-learn>=1.0.0
numpy>=1.21.0
```

## 🚀 Installation

### 1. Clone or Download the Project
```bash
# Download the project files to your local machine
```

### 2. Install Dependencies
Before installing, make sure your Python version is 3.10 or 3.11. Python 3.13 is not recommended due to package compatibility issues.

Install required packages:
```bash
pip install -r requirements.txt
```
If you face issues with NumPy or flwr, run this first:
```bash
pip install --upgrade pip setuptools wheel build
```

### 3. Verify Installation
```bash
python -c "import torch, transformers, flwr; print('All dependencies installed!')"
```

## 🎮 Usage

### Network Setup

**Important**: Both server and client machines must be on the same network.

1. **Find Server IP Address**:
   
   **Windows**:
   ```cmd
   ipconfig
   ```
   
   **macOS/Linux**:
   ```bash
   ifconfig | grep inet
   ```

2. **Update Server IP**: Edit the `server_ip` variable in `server.py`and `client.py` with your actual IP address.

### Running the System

#### Step 1: Start the Server

On the server machine:
```bash
python server.py
```

You should see output like:
```
Starting Flower Federated Learning Server...
============================================================
Server will run on: x.x.x.x:8081(Default port)
Make sure your client connects to this address!
============================================================
Server: Starting server and waiting for clients...
```

#### Step 2: Start Client(s)

On client machine(s):
```bash
python client.py
```

When prompted, enter the server address (or press Enter for default):
```
Enter server address (press Enter for x.x.x.x:8081): 
```

#### Step 3: Monitor Training

The federated learning will automatically start once clients connect. You'll see:

**Server Output**:
- Client connections
- Training round progress
- Aggregated metrics

**Client Output**:
- Model loading progress
- Training progress per batch
- Evaluation results

## 📁 Project Structure

```
flower_federated_albert/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── server.py                # Federated learning server
├── client.py                # Federated learning client
├── model.py                 # ALBERT model wrapper 

``

## 🔧 Configuration

### Server Configuration (in server.py)
```python
# Update this IP address to your server's IP
server_ip = "x.x.x.x"  # Change this to your IP
server_port = 8081
num_rounds = 3
```

### Model Configuration (in client.py)
```python
# Model settings
model_name = "albert-base-v2"
batch_size = 16
learning_rate = 5e-5
max_length = 128
dataset_split = "train[:2%]"  # 2% of IMDB dataset for demo
```

## 🌐 Network Configuration

### Same Network Setup (Recommended)
1. Connect all devices to the same WiFi network
2. Find server IP using `ipconfig` (Windows) or `ifconfig` (macOS/Linux)
3. Update `server_ip` in `server.py`
4. Start server first, then clients

### Local Testing
For testing on a single machine:
- Use `127.0.0.1:8081` as server address
- Run server and client in different terminals

## 📊 Training Details

### Dataset
- **Source**: IMDB movie reviews
- **Size**: 2% subset (500 samples total)
- **Split**: 70% training, 30% testing
- **Task**: Binary sentiment classification (positive/negative)

### Model
- **Architecture**: ALBERT-base-v2
- **Parameters**: ~12M parameters
- **Input**: Text sequences up to 128 tokens
- **Output**: Binary classification (2 classes)

### Training
- **Rounds**: 3 federated rounds
- **Local Epochs**: 1 per round
- **Batch Size**: 16
- **Learning Rate**: 5e-5
- **Optimizer**: Adam

## 🐛 Troubleshooting

### Common Issues

#### 1. Connection Timeout
```
grpc._channel._MultiThreadedRendezvous: ping timeout
```
**Solutions**:
- Verify server IP address is correct
- Ensure both machines are on same network
- Check firewall settings
- Try restarting both server and client

#### 2. Server Not Starting
```
OSError: [Errno 48] Address already in use
```
**Solutions**:
- Change port number in server.py
- Kill existing process using the port
- Wait a few minutes and try again

#### 3. Module Not Found
```
ModuleNotFoundError: No module named 'flwr'
```
**Solutions**:
- Install dependencies: `pip install -r requirements.txt`
- Check Python environment
- Verify Python version (3.8+)

#### 4. Model Download Issues
```
ConnectionError during model download
```
**Solutions**:
- Check internet connection
- Try running again (models are cached after first download)
- Use different network if corporate firewall blocks downloads

### Network Troubleshooting

#### Finding Your IP Address
**Windows**:
```cmd
ipconfig | findstr IPv4
```

**macOS**:
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

or

```bash
ipconfig getifaddr en0
```

**Linux**:
```bash
hostname -I
```

#### Testing Network Connectivity
```bash
# Test if server is reachable (replace with your server IP)
ping x.x.x.x

# Test if port is open (on client machine)
telnet x.x.x.x port
```

## 🔧 Customization

### Using Different Models
Replace `albert-base-v2` with other models:
```python
# In client.py
tokenizer = AlbertTokenizer.from_pretrained("distilbert-base-uncased")
model = AlbertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
```

### Adjusting Dataset Size
```python
# In client.py load_data() function
dataset = load_dataset("imdb", split="train[:10%]")  # Use 10% instead of 2%
```

### Changing Training Parameters
```python
# In client.py train() function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Lower learning rate
# In client.py DataLoader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)  # Smaller batch
```

## 📈 Expected Results

With the default configuration (2% IMDB dataset, 3 rounds):
- **Initial Accuracy**: ~50% (random)
- **After Round 1**: ~60-70%
- **After Round 3**: ~75-85%
- **Training Time**: 2-5 minutes per round (depending on hardware)

## 🤝 Multiple Clients

To run multiple clients:
1. Start server once
2. Run `python client.py` on multiple machines
3. Each client will train on the same dataset subset
4. Server aggregates results from all clients

## 📞 Support

### Before Asking for Help
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure network connectivity between machines
4. Check that IP addresses are correct

### Common Solutions
- Restart both server and client
- Check firewall settings
- Verify Python version compatibility
- Ensure sufficient memory/disk space

## 🙏 Acknowledgments

- [Flower Framework](https://flower.dev/) for federated learning
- [Hugging Face](https://huggingface.co/) for ALBERT model and datasets
- [PyTorch](https://pytorch.org/) for deep learning framework

---
