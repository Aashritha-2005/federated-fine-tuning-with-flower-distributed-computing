import flwr as fl
import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from typing import Dict, List, Tuple, Optional
import socket
import numpy as np

class FederatedStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(
            min_fit_clients=1,          
            min_evaluate_clients=1,     
            min_available_clients=1,    
            evaluate_metrics_aggregation_fn=self.weighted_average,
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.evaluate_config,
        )
        # Initialize the smallest ALBERT model - albert-base-v1
        print("Server: Loading albert-base-v1 model")
        self.model = AlbertForSequenceClassification.from_pretrained("albert-base-v1", num_labels=2)
        print("Server: Model loaded successfully")
        
        # Track rounds and results
        self.round_results = {}
        
    def initialize_parameters(self, client_manager) -> Optional[fl.common.Parameters]:
        """Initialize global model parameters."""
        print("Server: Initializing global model parameters...")
        try:
            # Get parameters from the ALBERT model
            parameters = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
            print(f"Server: Extracted {len(parameters)} parameter arrays")
            return fl.common.ndarrays_to_parameters(parameters)
        except Exception as e:
            print(f"Server: Error initializing parameters: {e}")
            raise
    
    def fit_config(self, server_round: int) -> Dict[str, str]:
        """Return training configuration dict for each round."""
        config = {
            "server_round": str(server_round),
            "local_epochs": "2",  # Increased local epochs for better training
            "batch_size": "4",
            "learning_rate": "1e-5"
        }
        print(f"Server: Sending fit config for round {server_round}: {config}")
        return config
    
    def evaluate_config(self, server_round: int) -> Dict[str, str]:
        """Return evaluation configuration dict for each round."""
        config = {
            "server_round": str(server_round),
        }
        print(f"Server: Sending evaluate config for round {server_round}: {config}")
        return config
    
    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate fit results using FedAvg weighted average."""
        print(f"\nServer: === ROUND {server_round} FIT AGGREGATION ===")
        print(f"Server: Received {len(results)} fit results, {len(failures)} failures")
        
        if failures:
            print("Server: Failures during fit:")
            for failure in failures:
                print(f"  - {failure}")
        
        if not results:
            print("Server: No fit results to aggregate!")
            return None, {}
        
        # Log client results and calculate training metrics
        total_examples = 0
        total_loss = 0
        client_info = []
        
        for i, (client_proxy, fit_res) in enumerate(results):
            client_examples = fit_res.num_examples
            client_loss = fit_res.metrics.get("train_loss", 0.0)
            client_id = fit_res.metrics.get("client_id", f"client_{i+1}")
            
            total_examples += client_examples
            total_loss += client_loss * client_examples
            
            client_info.append({
                "client_id": client_id,
                "examples": client_examples,
                "loss": client_loss
            })
            
            print(f"Server: {client_id} - Samples: {client_examples}, Loss: {client_loss:.4f}")
        
        # Calculate weighted average training loss
        avg_train_loss = total_loss / total_examples if total_examples > 0 else 0.0
        
        # Call parent aggregation (FedAvg)
        try:
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
            
            if aggregated_parameters is not None:
                # Add training metrics
                aggregated_metrics["train_loss"] = avg_train_loss
                aggregated_metrics["total_examples"] = total_examples
                aggregated_metrics["num_clients"] = len(results)
                
                print(f"Server: Successfully aggregated parameters for round {server_round}")
                print(f"Server: Round {server_round} Training Results:")
                print(f"  - Average Training Loss: {avg_train_loss:.4f}")
                print(f"  - Total Examples: {total_examples}")
                print(f"  - Number of Clients: {len(results)}")
                
                # Store round results
                self.round_results[f"round_{server_round}_fit"] = {
                    "train_loss": avg_train_loss,
                    "total_examples": total_examples,
                    "num_clients": len(results),
                    "client_info": client_info
                }
            else:
                print(f"Server: Failed to aggregate parameters for round {server_round}")
            
            return aggregated_parameters, aggregated_metrics
        except Exception as e:
            print(f"Server: Error during fit aggregation: {e}")
            raise
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation results."""
        print(f"\nServer: === ROUND {server_round} EVALUATE AGGREGATION ===")
        print(f"Server: Received {len(results)} evaluate results, {len(failures)} failures")
        
        if failures:
            print("Server: Failures during evaluate:")
            for failure in failures:
                print(f"  - {failure}")
        
        if not results:
            print("Server: No evaluate results to aggregate!")
            return None, {}
        
        # Log client results and calculate evaluation metrics
        total_examples = 0
        total_loss = 0
        total_accuracy = 0
        client_eval_info = []
        
        for i, (client_proxy, evaluate_res) in enumerate(results):
            client_examples = evaluate_res.num_examples
            client_loss = evaluate_res.loss
            client_accuracy = evaluate_res.metrics.get("eval_accuracy", 0.0)
            client_id = evaluate_res.metrics.get("client_id", f"client_{i+1}")
            
            total_examples += client_examples
            total_loss += client_loss * client_examples
            total_accuracy += client_accuracy * client_examples
            
            client_eval_info.append({
                "client_id": client_id,
                "examples": client_examples,
                "loss": client_loss,
                "accuracy": client_accuracy
            })
            
            print(f"Server: {client_id} - Loss: {client_loss:.4f}, Accuracy: {client_accuracy:.4f}, Samples: {client_examples}")
        
        # Calculate weighted averages
        avg_eval_loss = total_loss / total_examples if total_examples > 0 else 0.0
        avg_eval_accuracy = total_accuracy / total_examples if total_examples > 0 else 0.0
        
        # Call parent aggregation
        try:
            aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
            
            # Add comprehensive evaluation metrics
            final_metrics = {
                "accuracy": avg_eval_accuracy,
                "eval_loss": avg_eval_loss,
                "total_examples": total_examples,
                "num_clients": len(results)
            }
            
            print(f"Server: Round {server_round} Evaluation Results:")
            print(f"  - Average Evaluation Loss: {avg_eval_loss:.4f}")
            print(f"  - Average Evaluation Accuracy: {avg_eval_accuracy:.4f}")
            print(f"  - Total Examples: {total_examples}")
            print(f"  - Number of Clients: {len(results)}")
            
            # Store round results
            self.round_results[f"round_{server_round}_eval"] = {
                "eval_loss": avg_eval_loss,
                "eval_accuracy": avg_eval_accuracy,
                "total_examples": total_examples,
                "num_clients": len(results),
                "client_info": client_eval_info
            }
            
            return aggregated_loss, final_metrics
        except Exception as e:
            print(f"Server: Error during evaluate aggregation: {e}")
            raise
    
    def weighted_average(self, metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        """Aggregate evaluation metrics from clients using weighted average."""
        if not metrics:
            print("Server: No metrics to aggregate")
            return {}
            
        print(f"Server: Aggregating metrics from {len(metrics)} clients")
        
        total_examples = sum([num_examples for num_examples, _ in metrics])
        print(f"Server: Total examples across all clients: {total_examples}")
        
        if total_examples == 0:
            return {}
        
        # Weighted average of accuracies and losses
        weighted_accuracy = 0.0
        weighted_loss = 0.0
        
        for num_examples, m in metrics:
            weight = num_examples / total_examples
            acc = m.get("eval_accuracy", 0.0)
            loss = m.get("eval_loss", 0.0)
            
            weighted_accuracy += weight * acc
            weighted_loss += weight * loss
            
            client_id = m.get("client_id", "unknown")
            print(f"Server: {client_id} - Weight: {weight:.3f}, Accuracy: {acc:.4f}, Loss: {loss:.4f}")
        
        final_metrics = {
            "accuracy": weighted_accuracy,
            "eval_loss": weighted_loss,
            "total_examples": total_examples
        }
        
        print(f"Server: Final aggregated metrics: {final_metrics}")
        return final_metrics
    
    def print_final_summary(self):
        """Print a summary of all rounds."""
        print("\n" + "=" * 80)
        print("FEDERATED LEARNING SUMMARY")
        print("=" * 80)
        
        for round_key in sorted(self.round_results.keys()):
            round_data = self.round_results[round_key]
            print(f"\n{round_key.upper()}:")
            
            if "fit" in round_key:
                print(f"  Training Loss: {round_data['train_loss']:.4f}")
                print(f"  Total Examples: {round_data['total_examples']}")
                print(f"  Number of Clients: {round_data['num_clients']}")
            else:  # eval
                print(f"  Evaluation Loss: {round_data['eval_loss']:.4f}")
                print(f"  Evaluation Accuracy: {round_data['eval_accuracy']:.4f}")
                print(f"  Total Examples: {round_data['total_examples']}")
                print(f"  Number of Clients: {round_data['num_clients']}")

def main():
    print("Starting Flower Federated Learning Server...")
    print("=" * 60)
    
    # Use the specified IP and port
    server_address = "192.168.33.72:8081"
    
    print(f"Server will run on: {server_address}")
    print("Server will start training after the first client connects!")
    print("Multiple clients can join during the process!")
    print("=" * 60)
    
    # Create strategy
    try:
        print("Server: Creating FedAvg strategy...")
        strategy = FederatedStrategy()
        print("Server: FedAvg strategy created successfully")
    except Exception as e:
        print(f"Server: Error creating strategy: {e}")
        return
    
    try:
        # Start server
        print("Server: Starting server and waiting for clients...")
        print("Server: Federated learning will begin once a client connects...")
        print("=" * 60)
        
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=3), 
            strategy=strategy,
        )
        
        print("=" * 60)
        print("Server: Federated learning completed successfully!")
        
        # Print final summary
        strategy.print_final_summary()
        
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Server: Stopped by user.")
        if hasattr(strategy, 'round_results') and strategy.round_results:
            strategy.print_final_summary()
    except Exception as e:
        print("=" * 60)
        print(f"Server: Error: {e}")
        print("Server: Make sure the IP address and port are available.")
        print("Server: Check if firewall is blocking the connection.")

if __name__ == "__main__":
    main()