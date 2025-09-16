import flwr as fl
import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import sys
import random
import time
import os
import socket

# Set random seeds for reproducibility but make each client unique
CLIENT_SEED = int(time.time() * 1000) % 10000
random.seed(CLIENT_SEED)
torch.manual_seed(CLIENT_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Client: Using device: {DEVICE}")

# Load the smallest ALBERT model available - albert-base-v1
print("Client: Loading albert-base-v1 tokenizer and model (smallest available)...")
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")
model = AlbertForSequenceClassification.from_pretrained("albert-base-v1", num_labels=2)
model.to(DEVICE)
print("Client: Model loaded and moved to device")

# Generate unique client ID with hostname for multi-laptop identification
try:
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    CLIENT_ID = f"client_{hostname}{local_ip.split('.')[-1]}{random.randint(1000, 9999)}"
except:
    CLIENT_ID = f"client_{random.randint(1000, 9999)}_{int(time.time() % 10000)}"

print(f"Client: Unique client ID: {CLIENT_ID}")

# Load small subset of dataset with client-specific split
def load_data():
    print("Client: Loading IMDB dataset...")
    # Use very small dataset for faster processing with multiple clients
    dataset = load_dataset("imdb", split="train[:100]")  # Small dataset (0.5%)
    
    # Create client-specific data split based on client ID
    total_size = len(dataset)
    client_seed = hash(CLIENT_ID) % 1000
    
    # Split data differently for each client to simulate real federated scenario
    dataset = dataset.train_test_split(test_size=0.3, seed=client_seed)
    
    print(f"Client {CLIENT_ID}: Loaded {len(dataset['train'])} training samples and {len(dataset['test'])} test samples")
    return dataset["train"], dataset["test"]

train_data, test_data = load_data()

# Tokenization with very small max length for efficiency
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)  # Small sequence length

print("Client: Tokenizing datasets...")
train_data = train_data.map(tokenize, batched=True)
test_data = test_data.map(tokenize, batched=True)

train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Small batch sizes for multiple clients
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4)
print("Client: Data loaders created with small batch sizes")

# Training function with memory optimization
def train(model, loader, config):
    print(f"Client {CLIENT_ID}: Starting training...")
    model.train()
    
    # Use small learning rate for stability with multiple clients
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Small learning rate
    total_loss = 0
    num_batches = 0
    
    epochs = int(config.get('local_epochs', 1))
    for epoch in range(epochs):
        print(f"Client {CLIENT_ID}: Training epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in enumerate(loader):
            try:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1
                
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                # Clear cache frequently
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    print(f"Client {CLIENT_ID}: Epoch {epoch+1}, Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"Client {CLIENT_ID}: Error in training batch {batch_idx}: {e}")
                continue
    
    avg_loss = total_loss / max(num_batches, 1)
    print(f"Client {CLIENT_ID}: Training completed. Average loss: {avg_loss:.4f}")
    return avg_loss

# Evaluation function with memory optimization
def test(model, loader, config):
    print(f"Client {CLIENT_ID}: Starting evaluation...")
    model.eval()
    predictions, labels = [], []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            try:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                batch_labels = batch["label"]
                labels.extend(batch_labels.numpy())
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels.to(DEVICE))
                total_loss += outputs.loss.item()
                num_batches += 1
                
                preds = outputs.logits.argmax(dim=1).cpu().numpy()
                predictions.extend(preds)
                
                # Clear cache frequently
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    print(f"Client {CLIENT_ID}: Evaluation batch {batch_idx}/{len(loader)}")
                    
            except Exception as e:
                print(f"Client {CLIENT_ID}: Error in evaluation batch {batch_idx}: {e}")
                continue
    
    acc = accuracy_score(labels, predictions) if len(labels) > 0 and len(predictions) > 0 else 0.0
    avg_loss = total_loss / max(num_batches, 1)
    print(f"Client {CLIENT_ID}: Evaluation completed. Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
    return avg_loss, acc

# Flower Client with better error handling
class ALBERTClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        print(f"Client {CLIENT_ID}: Getting model parameters...")
        try:
            parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
            print(f"Client {CLIENT_ID}: Extracted {len(parameters)} parameter arrays")
            return parameters
        except Exception as e:
            print(f"Client {CLIENT_ID}: Error getting parameters: {e}")
            raise

    def set_parameters(self, parameters):
        print(f"Client {CLIENT_ID}: Setting model parameters...")
        try:
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            print(f"Client {CLIENT_ID}: Successfully set {len(parameters)} parameter arrays")
        except Exception as e:
            print(f"Client {CLIENT_ID}: Error setting parameters: {e}")
            raise

    def fit(self, parameters, config):
        server_round = config.get('server_round', 'unknown')
        print(f"Client {CLIENT_ID}: === STARTING FIT ROUND {server_round} ===")
        
        try:
            self.set_parameters(parameters)
            avg_loss = train(model, train_loader, config)
            updated_parameters = self.get_parameters(config={})
            
            metrics = {
                "train_loss": float(avg_loss), 
                "client_id": CLIENT_ID,
                "num_examples": len(train_loader.dataset)
            }
            print(f"Client {CLIENT_ID}: Fit round {server_round} completed successfully")
            print(f"Client {CLIENT_ID}: Returning {len(train_loader.dataset)} samples with metrics: {metrics}")
            
            return updated_parameters, len(train_loader.dataset), metrics
        except Exception as e:
            print(f"Client {CLIENT_ID}: Error during fit round {server_round}: {e}")
            raise

    def evaluate(self, parameters, config):
        server_round = config.get('server_round', 'unknown')
        print(f"Client {CLIENT_ID}: === STARTING EVALUATE ROUND {server_round} ===")
        
        try:
            self.set_parameters(parameters)
            avg_loss, acc = test(model, test_loader, config)
            
            metrics = {
                "eval_accuracy": float(acc), 
                "eval_loss": float(avg_loss),
                "client_id": CLIENT_ID,
                "num_examples": len(test_loader.dataset)
            }
            print(f"Client {CLIENT_ID}: Evaluate round {server_round} completed successfully")
            print(f"Client {CLIENT_ID}: Returning loss: {avg_loss:.4f}, accuracy: {acc:.4f}, samples: {len(test_loader.dataset)}")
            
            return float(avg_loss), len(test_loader.dataset), metrics
        except Exception as e:
            print(f"Client {CLIENT_ID}: Error during evaluate round {server_round}: {e}")
            raise

def test_server_connectivity(server_address):
    """Test if the server is reachable."""
    try:
        host, port = server_address.split(':')
        port = int(port)
        
        print(f"Client {CLIENT_ID}: Testing connectivity to {server_address}...")
        
        # Create a socket and try to connect
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.settimeout(10)  # 10 second timeout
        result = test_socket.connect_ex((host, port))
        test_socket.close()
        
        if result == 0:
            print(f"Client {CLIENT_ID}: Successfully connected to server!")
            return True
        else:
            print(f"Client {CLIENT_ID}: Cannot connect to server (error code: {result})")
            return False
    except Exception as e:
        print(f"Client {CLIENT_ID}: Connectivity test failed: {e}")
        return False

def get_client_info():
    """Get client system information for debugging."""
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"Client {CLIENT_ID}: Running on hostname: {hostname}")
        print(f"Client {CLIENT_ID}: Local IP address: {local_ip}")
    except Exception as e:
        print(f"Client {CLIENT_ID}: Could not get system info: {e}")

def main():
    print(f"Starting Flower Federated Learning Client for Multi-Laptop Setup...")
    print("=" * 70)
    
    # Get client system information
    get_client_info()
    
    # Server address for multi-laptop setup
    server_address = "192.168.33.72:8081"
    
    print(f"Client {CLIENT_ID}: Connecting to server at: {server_address}")
    print("Client: Make sure you're connected to the same mobile hotspot as the server!")
    print("Client: Make sure the server is running before starting the client!")
    print("=" * 70)
    
    # Test connectivity first
    if not test_server_connectivity(server_address):
        print("=" * 70)
        print("CONNECTIVITY TROUBLESHOOTING:")
        print("1. Ensure both client and server are on the same mobile hotspot")
        print("2. Verify the server is running and listening on 192.168.33.72:8081")
        print("3. Check if firewall is blocking the connection")
        print("4. Try pinging the server: ping 192.168.33.72")
        print("=" * 70)
        return
    
    # Connection retry logic with longer timeouts for network delays
    max_retries = 2
    retry_delay = 10  # Start with 10 seconds
    
    for attempt in range(max_retries):
        try:
            print(f"Client {CLIENT_ID}: Creating client instance...")
            client = ALBERTClient()
            print(f"Client {CLIENT_ID}: Client instance created successfully")
            
            print(f"Client {CLIENT_ID}: Attempting connection to server (attempt {attempt + 1}/{max_retries})...")
            
            # Start the Flower client with longer timeout for network operations
            fl.client.start_numpy_client(
                server_address=server_address, 
                client=client,
                grpc_max_message_length=1024*1024*1024,  # 1GB max message size
            )
            
            print("=" * 70)
            print(f"Client {CLIENT_ID}: Federated learning completed successfully!")
            break
            
        except KeyboardInterrupt:
            print(f"\n" + "=" * 70)
            print(f"Client {CLIENT_ID}: Stopped by user.")
            break
        except Exception as e:
            print("=" * 70)
            print(f"Client {CLIENT_ID}: Connection error (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                print(f"Client {CLIENT_ID}: Retrying in {retry_delay} seconds...")
                print("Client: Make sure the server is still running...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.2, 60)  # Exponential backoff, max 60 seconds
            else:
                print(f"Client {CLIENT_ID}: All connection attempts failed.")
                print("=" * 70)
                print("FINAL TROUBLESHOOTING STEPS:")
                print("1. Verify server is running on the server laptop")
                print("2. Check network connectivity between laptops")
                print("3. Ensure both devices are on the same mobile hotspot")
                print("4. Try restarting both server and client")
                print(f"5. Server should be accessible at: {server_address}")

if __name__ == "__main__":
    main()
