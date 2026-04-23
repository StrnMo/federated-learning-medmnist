"""
Federated learning experiment on MedMNIST dataset.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.medmnist_loader import load_medmnist
from utils.data_split import create_non_iid_splits
from models.cnn_model import get_model
from federated.fedavg import run_fedavg, evaluate_global_model

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_NAME = 'breastmnist'  # Change as needed
DATA_ROOT = 'C:/Users/ASUS/.medmnist'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLIENTS = 5
NUM_ROUNDS = 20
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
SAMPLES_PER_CLIENT = 1500

def run_federated_experiment():
    print("=" * 60)
    print("FEDERATED LEARNING EXPERIMENT")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Number of clients: {NUM_CLIENTS}")
    print(f"Number of rounds: {NUM_ROUNDS}")
    print(f"Local epochs: {LOCAL_EPOCHS}")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading MedMNIST dataset...")
    train_dataset, val_dataset, test_dataset, num_classes = load_medmnist(DATASET_NAME, download=False,root=DATA_ROOT)
    
    # Create test loader for evaluation
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Create non-IID client splits
    print("\n2. Creating non-IID client splits...")
    client_indices = create_non_iid_splits(
        train_dataset, 
        num_clients=NUM_CLIENTS, 
        samples_per_client=SAMPLES_PER_CLIENT
    )
    
    # Prepare clients data for federated learning
    clients_data = [(train_dataset, indices) for indices in client_indices]
    
    # Initialize global model
    print("\n3. Initializing global model...")
    global_model = get_model(num_classes).to(DEVICE)
    
    # Run centralized baseline (optional, for comparison)
    print("\n4. Optional: Running centralized baseline (quick version)...")
    from experiments.run_centralized import run_centralized
    centralized_model, centralized_acc, _, _ = run_centralized()
    
    # Run Federated Learning
    print("\n5. Running Federated Learning...")
    trained_model, history = run_fedavg(
        clients_data=clients_data,
        global_model=global_model,
        device=DEVICE,
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        test_loader=test_loader,
        centralized_acc=centralized_acc
    )
    
    # Final evaluation
    final_acc = evaluate_global_model(trained_model, test_loader, DEVICE)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED")
    print("=" * 60)
    print(f"Centralized Test Accuracy: {centralized_acc:.4f}")
    print(f"Federated Final Test Accuracy: {final_acc:.4f}")
    print(f"Performance Gap: {centralized_acc - final_acc:.4f}")
    print("=" * 60)
    
    # Plot results
    plot_results(history, centralized_acc, centralized_acc)

def plot_results(history, centralized_acc, final_acc):
    """Plot FL training results."""
    plt.figure(figsize=(12, 5))
    
    # Accuracy over rounds
    plt.subplot(1, 2, 1)
    plt.plot(history['rounds'], history['test_accuracies'], 'b-o', label='Federated Learning')
    plt.axhline(y=centralized_acc, color='r', linestyle='--', label=f'Centralized ({centralized_acc:.3f})')
    plt.xlabel('Communication Round')
    plt.ylabel('Test Accuracy')
    plt.title('Federated vs Centralized Learning')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Round time
    plt.subplot(1, 2, 2)
    plt.bar(history['rounds'], history['times'], color='green', alpha=0.7)
    plt.xlabel('Communication Round')
    plt.ylabel('Time per Round (seconds)')
    plt.title('Training Time per FL Round')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Federated Learning on MedMNIST ({NUM_CLIENTS} clients, {NUM_ROUNDS} rounds)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/fl_medmnist_results.png', dpi=150)
    plt.savefig('results/fl_medmnist_results.pdf')
    print("\n✅ Plot saved to results/fl_medmnist_results.png")

if __name__ == "__main__":
    # Create results folder if it doesn't exist
    os.makedirs('results', exist_ok=True)
    run_federated_experiment()