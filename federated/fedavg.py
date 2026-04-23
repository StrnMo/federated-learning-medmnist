"""
Main FedAvg training loop.
"""

import copy
import time
import torch
import numpy as np
from federated.client import FLClient
from federated.server import FLServer

def run_fedavg(clients_data, global_model, device, num_rounds=20, local_epochs=3, 
               test_loader=None, centralized_acc=None):
    """
    Execute FedAvg algorithm.
    
    Args:
        clients_data: list of (dataset, indices) tuples for each client
        global_model: initialized model
        device: 'cuda' or 'cpu'
        num_rounds: number of communication rounds
        local_epochs: number of local training epochs per round
        test_loader: DataLoader for evaluation
        centralized_acc: Centralized accuracy for comparison
    
    Returns:
        trained global model, history
    """
    print("\n" + "=" * 60)
    print("STARTING FEDERATED LEARNING (FedAvg)")
    print("=" * 60)
    print(f"Number of clients: {len(clients_data)}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Local epochs per round: {local_epochs}")
    print("=" * 60)
    
    # Initialize clients
    clients = []
    for i, (dataset, indices) in enumerate(clients_data):
        client_model = copy.deepcopy(global_model)
        client = FLClient(i, indices, dataset, client_model, device)
        clients.append(client)
    
    # Initialize server
    server = FLServer(copy.deepcopy(global_model), device)
    
    # History tracking
    history = {
        'rounds': [],
        'train_losses': [],
        'test_accuracies': [],
        'times': []
    }
    
    start_time = time.time()
    
    for round_idx in range(num_rounds):
        round_start = time.time()
        print(f"\n--- Round {round_idx + 1}/{num_rounds} ---")
        
        # Distribute global weights to all clients
        global_weights = server.get_weights()
        client_weights = []
        
        for client in clients:
            client.set_weights(copy.deepcopy(global_weights))
            weights = client.train(local_epochs=local_epochs)
            client_weights.append(weights)
        
        # Aggregate weights using FedAvg
        server.aggregate_fedavg(client_weights)
        
        # Evaluate if test_loader is provided
        if test_loader:
            acc = evaluate_global_model(server.global_model, test_loader, device)
            history['test_accuracies'].append(acc)
            
            # Show comparison with centralized baseline
            if centralized_acc:
                gap = centralized_acc - acc
                print(f"  Round {round_idx + 1} complete. Test Acc: {acc:.4f} (Gap: {gap:.4f})")
            else:
                print(f"  Round {round_idx + 1} complete. Test Acc: {acc:.4f}")
        else:
            print(f"  Round {round_idx + 1} complete")
        
        history['rounds'].append(round_idx + 1)
        history['times'].append(time.time() - round_start)
    
    total_time = time.time() - start_time
    print(f"\n" + "=" * 60)
    print(f"FEDERATED LEARNING COMPLETED in {total_time/60:.1f} minutes")
    print("=" * 60)
    
    return server.global_model, history

def evaluate_global_model(model, test_loader, device):
    """Evaluate global model on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total if total > 0 else 0