"""
Centralized baseline training on MedMNIST dataset.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.medmnist_loader import load_medmnist
from models.cnn_model import get_model

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_NAME = 'breastmnist'  # Change to 'chestmnist', 'dermamnist', etc.
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
DATA_ROOT = 'C:/Users/ASUS/.medmnist'


def run_centralized():
    print("=" * 60)
    print("CENTRALIZED BASELINE TRAINING")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading MedMNIST dataset...")
    train_dataset, val_dataset, test_dataset, num_classes = load_medmnist(DATASET_NAME, download=False,root=DATA_ROOT)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    print("\n2. Initializing model...")
    model = get_model(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\n3. Training centralized model...")
    train_losses = []
    val_accuracies = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.squeeze().long().to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.squeeze().long().to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        val_accuracies.append(val_acc)
        
        print(f"  Epoch {epoch+1}/{EPOCHS}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")
    
    # Final test evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.squeeze().long().to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = correct / total
    
    print("\n" + "=" * 60)
    print("CENTRALIZED TRAINING COMPLETED")
    print("=" * 60)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Validation Accuracies: {val_accuracies}")
    print("=" * 60)
    
    return model, test_acc, train_losses, val_accuracies

if __name__ == "__main__":
    run_centralized()