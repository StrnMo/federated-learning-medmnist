"""
Non-IID data splitting for federated learning clients.
Creates realistic hospital-like data heterogeneity.
"""

import numpy as np
from torch.utils.data import Subset

def create_non_iid_splits(dataset, num_clients=5, samples_per_client=1500, alpha=0.5):
    """
    Create non-IID splits using Dirichlet distribution.
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        samples_per_client: Maximum samples per client
        alpha: Dirichlet concentration parameter (smaller = more non-IID)
    
    Returns:
        client_indices: List of index lists for each client
    """
    # Get labels
    labels = np.array([label for _, label in dataset])
    num_classes = len(np.unique(labels))
    
    # Get indices per class
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    # Distribute samples using Dirichlet distribution
    client_indices = [[] for _ in range(num_clients)]
    
    for class_id in range(num_classes):
        # Generate proportions for this class across clients
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # Assign samples to clients
        class_samples = class_indices[class_id]
        np.random.shuffle(class_samples)
        
        start = 0
        for client_id, prop in enumerate(proportions):
            n_samples = int(prop * len(class_samples))
            if n_samples > 0:
                end = start + n_samples
                client_indices[client_id].extend(class_samples[start:end])
                start = end
    
    # Shuffle each client's indices and limit samples
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        if len(client_indices[i]) > samples_per_client:
            client_indices[i] = client_indices[i][:samples_per_client]
    
    # Print distribution statistics
    print(f"\nClient distribution ({num_clients} clients):")
    for i, indices in enumerate(client_indices):
        client_labels = labels[indices]
        unique, counts = np.unique(client_labels, return_counts=True)
        print(f"  Client {i}: {len(indices)} samples, classes: {dict(zip(unique, counts))}")
    
    return client_indices

def create_iid_splits(dataset, num_clients=5, samples_per_client=1500):
    """Create IID splits (uniform random sampling)."""
    total_samples = len(dataset)
    indices = list(range(total_samples))
    np.random.shuffle(indices)
    
    client_indices = []
    for i in range(num_clients):
        start = i * samples_per_client
        end = min(start + samples_per_client, total_samples)
        client_indices.append(indices[start:end])
    
    return client_indices

if __name__ == "__main__":
    # Test the splitter
    from medmnist_loader import load_medmnist
    
    train, _, _, _ = load_medmnist('pathmnist')
    indices = create_non_iid_splits(train, num_clients=5, samples_per_client=1000)
    print(f"\nCreated {len(indices)} clients successfully.")