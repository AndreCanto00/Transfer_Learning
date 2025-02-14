import pytest
import torch
from src.data.data_loader import get_data_transforms, create_data_loaders
from torchvision import datasets
import os
from PIL import Image
import numpy as np

@pytest.fixture
def sample_image_folders(tmp_path):
    """Create sample image folders with dummy images for testing."""
    # Create directories for train, val, test
    paths = {}
    for split in ['train', 'val', 'test']:
        split_path = tmp_path / split
        class_path = split_path / 'class1'
        os.makedirs(class_path)
        
        # Create multiple dummy images to satisfy batch size
        for i in range(3):  # Create 3 images
            img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
            img.save(class_path / f'sample_{i}.jpg')
        
        paths[split] = split_path
    
    return paths

def test_get_data_transforms():
    """Test data transformation pipeline."""
    transforms = get_data_transforms()
    
    # Create a dummy image
    img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    
    # Apply transforms
    transformed_img = transforms(img)
    
    # Check output properties
    assert isinstance(transformed_img, torch.Tensor)
    assert transformed_img.shape == (3, 224, 224)
    assert transformed_img.dtype == torch.float32

def test_create_data_loaders(sample_image_folders):
    """Test creation of data loaders."""
    train_loader, val_loader, test_loader = create_data_loaders(
        sample_image_folders['train'],
        sample_image_folders['val'],
        sample_image_folders['test'],
        batch_size=2
    )
    
    # Check loader properties
    assert train_loader.batch_size == 2
    assert val_loader.batch_size == 2
    assert test_loader.batch_size == 2
    
    # Check if loaders contain data
    train_batch = next(iter(train_loader))
    assert len(train_batch) == 2  # (images, labels)
    assert train_batch[0].shape == (2, 3, 224, 224)  # batch_size x channels x height x width