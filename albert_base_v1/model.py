import os
import torch
import torch.nn as nn
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from typing import List
import numpy as np

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

class AlbertClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Use the smallest ALBERT model available - albert-base-v1
        self.model_name = "albert-base-v1"
        self.albert = AlbertForSequenceClassification.from_pretrained(self.model_name, num_labels=num_classes)
        print(f"Model: Loaded {self.model_name} with {num_classes} classes")

    def forward(self, input_ids, attention_mask, labels=None):
        return self.albert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

class ModelManager:
    def __init__(self, num_classes=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ModelManager: Using device: {self.device}")
        
        # Initialize the smallest ALBERT model
        self.model = AlbertClassifier(num_classes=num_classes)
        self.model.to(self.device)
        
        # Initialize tokenizer
        self.tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")
        print("ModelManager: Model and tokenizer initialized successfully")

    def get_parameters(self) -> List[np.ndarray]:
        """Extract model parameters as numpy arrays."""
        print("ModelManager: Extracting model parameters...")
        parameters = [param.detach().cpu().numpy() for param in self.model.parameters()]
        print(f"ModelManager: Extracted {len(parameters)} parameter arrays")
        return parameters

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters from numpy arrays."""
        print("ModelManager: Setting model parameters...")
        try:
            for param, new_param in zip(self.model.parameters(), parameters):
                param.data = torch.tensor(new_param, dtype=param.dtype, device=self.device)
            print("ModelManager: Parameters set successfully")
        except Exception as e:
            print(f"ModelManager: Error setting parameters: {e}")
            raise

    def tokenize_data(self, texts, max_length=64):
        """Tokenize input texts with smaller max_length for efficiency."""
        return self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )

    def get_model_info(self):
        """Get information about the model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            "model_name": self.model.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device)
        }
        
        print(f"ModelManager: Model info - {info}")
        return info

def create_sample_dataset():
    """Create a small sample dataset for testing."""
    texts = [
        "This movie is amazing and fantastic!", 
        "Loved the direction, acting and story.",
        "Great cinematography and excellent plot.",
        "Wonderful performances by all actors.",
        "Worst plot ever seen in cinema.", 
        "Absolutely terrible and boring film.",
        "Poor direction and bad acting throughout.",
        "Disappointing story with no substance."
    ]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]  # 1 for positive, 0 for negative
    return texts, labels

class FlowerDataset(torch.utils.data.Dataset):
    """Custom dataset class for Flower federated learning."""
    def __init__(self, texts, labels, tokenizer, max_length=64):
        print(f"FlowerDataset: Creating dataset with {len(texts)} samples")
        self.encodings = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        self.labels = labels
        print(f"FlowerDataset: Dataset created successfully")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Utility functions for federated learning
def create_federated_datasets(num_clients=3, dataset_size_per_client=50):
    """Create federated datasets for multiple clients."""
    print(f"Creating federated datasets for {num_clients} clients...")
    
    # Create larger sample dataset
    base_texts, base_labels = create_sample_dataset()
    
    # Replicate data to create larger dataset
    texts = base_texts * (dataset_size_per_client // len(base_texts) + 1)
    labels = base_labels * (dataset_size_per_client // len(base_labels) + 1)
    
    # Trim to exact size
    texts = texts[:dataset_size_per_client * num_clients]
    labels = labels[:dataset_size_per_client * num_clients]
    
    # Split into client datasets
    client_datasets = []
    for i in range(num_clients):
        start_idx = i * dataset_size_per_client
        end_idx = (i + 1) * dataset_size_per_client
        
        client_texts = texts[start_idx:end_idx]
        client_labels = labels[start_idx:end_idx]
        
        client_datasets.append((client_texts, client_labels))
        print(f"Client {i+1}: {len(client_texts)} samples")
    
    return client_datasets

def evaluate_global_model(model_manager, test_texts, test_labels):
    """Evaluate the global model on test data."""
    print("Evaluating global model...")
    
    model_manager.model.eval()
    correct = 0
    total = 0
    total_loss = 0
    
    # Tokenize test data
    encodings = model_manager.tokenize_data(test_texts)
    
    with torch.no_grad():
        for i in range(len(test_labels)):
            input_ids = encodings["input_ids"][i:i+1].to(model_manager.device)
            attention_mask = encodings["attention_mask"][i:i+1].to(model_manager.device)
            labels = torch.tensor([test_labels[i]], dtype=torch.long).to(model_manager.device)
            
            outputs = model_manager.model(input_ids, attention_mask, labels)
            loss = outputs.loss
            predictions = outputs.logits.argmax(dim=1)
            
            total_loss += loss.item()
            total += 1
            correct += (predictions == labels).sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    
    print(f"Global model evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
