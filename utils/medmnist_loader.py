"""
Data loader for MedMNIST medical image datasets.
Supports BreastMNIST with automatic tensor conversion.
"""

import medmnist
from medmnist import INFO
import torch
import torchvision.transforms as transforms
import os

# Define the transform to convert PIL images to tensors
transform = transforms.Compose([
    transforms.ToTensor(),
])

class BreastMNISTWithTransform(medmnist.BreastMNIST):
    def __init__(self, split='train', download=True, size=28, root=None):
        # Ensure root is passed properly
        if root is not None:
            super().__init__(split=split, download=download, size=size, root=root)
        else:
            super().__init__(split=split, download=download, size=size)
        self.transform = transform

class PathMNISTWithTransform(medmnist.PathMNIST):
    def __init__(self, split='train', download=True, size=28, root=None):
        if root is not None:
            super().__init__(split=split, download=download, size=size, root=root)
        else:
            super().__init__(split=split, download=download, size=size)
        self.transform = transform

DATASET_CLASSES = {
    'breastmnist': BreastMNISTWithTransform,
    'pathmnist': PathMNISTWithTransform,
}

DATASET_INFO = {
    'breastmnist': 'BreastMNIST - Breast Ultrasound (2 classes)',
    'pathmnist': 'PathMNIST - Colon Pathology (9 classes)',
}

def load_medmnist(dataset_name='breastmnist', download=False, size=28, root=None):
    """
    Load MedMNIST dataset with proper tensor conversion.
    
    Args:
        dataset_name: 'breastmnist', 'pathmnist'
        download: Set to False to use local file
        size: Image size (28 is standard)
        root: Custom root directory for dataset storage
    
    Returns:
        train_dataset, val_dataset, test_dataset, num_classes
    """
    if dataset_name.lower() not in DATASET_CLASSES:
        raise ValueError(f"Unknown dataset. Choose from: {list(DATASET_CLASSES.keys())}")
    
    print(f"Loading {DATASET_INFO.get(dataset_name, dataset_name)}...")
    print(f"Image size: {size}x{size}")
    if root:
        print(f"Data root: {root}")
    
    # Get the dataset class
    dataset_class = DATASET_CLASSES[dataset_name.lower()]
    
    # Load datasets with explicit root
    train_dataset = dataset_class(split='train', download=download, size=size, root=root)
    val_dataset = dataset_class(split='val', download=download, size=size, root=root)
    test_dataset = dataset_class(split='test', download=download, size=size, root=root)
    
    # Get number of classes from INFO
    info = INFO[dataset_name.lower()]
    num_classes = len(info['label'])
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Number of classes: {num_classes}")
    
    return train_dataset, val_dataset, test_dataset, num_classes

if __name__ == "__main__":
    print("=" * 50)
    print("Testing BreastMNIST Loader")
    print("=" * 50)
    
    train, val, test, n_classes = load_medmnist('breastmnist', download=False, root='C:/Users/ASUS/.medmnist')
    
    print("=" * 50)
    print("SUCCESS! Loader works correctly.")
    print("=" * 50)
    
    sample_img, sample_label = train[0]
    print(f"\nSample image shape: {sample_img.shape}")
    print(f"Sample label: {sample_label}")
    print("=" * 50)