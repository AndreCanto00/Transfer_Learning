import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_data_transforms(input_size=(224, 224)):
    """
    Get standard data transformations for image preprocessing.
    
    Args:
        input_size (tuple): Desired input size (height, width)
    
    Returns:
        transforms.Compose: Composition of transforms
    """
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def create_data_loaders(train_path, val_path, test_path, 
                       batch_size=32, num_workers=4):
    """
    Create DataLoader objects for train, validation and test sets.
    
    Args:
        train_path (str): Path to training data
        val_path (str): Path to validation data
        test_path (str): Path to test data
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of workers for DataLoader
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    transform = get_data_transforms()
    
    # Create datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader