"""
Federated learning client: trains on local medical data.
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

class FLClient:
    def __init__(self, client_id, indices, dataset, model, device, batch_size=64):
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device
        self.dataloader = DataLoader(
            Subset(dataset, indices), 
            batch_size=batch_size, 
            shuffle=True
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, local_epochs=3, lr=0.001):
        """Train client model locally."""
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        total_loss = 0
        num_batches = 0
        
        for epoch in range(local_epochs):
            epoch_loss = 0
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.squeeze().long().to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                total_loss += loss.item()
                num_batches += 1
            
            if local_epochs > 1:
                print(f"      Client {self.client_id} - Epoch {epoch+1}: Loss {epoch_loss/len(self.dataloader):.4f}")
        
        # Return model weights
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
    
    def set_weights(self, weights):
        """Set model weights from global model."""
        self.model.load_state_dict(weights)
        self.model.to(self.device)
    
    def evaluate(self, test_loader):
        """Evaluate client model on test set."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.squeeze().long().to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total if total > 0 else 0