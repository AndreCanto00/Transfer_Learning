import os
import pytest
import shutil
from src.data.dataset_utils import split_dataset

@pytest.fixture
def sample_dataset(tmp_path):
    """Create a sample dataset structure for testing."""
    # Create class directories
    class1_dir = tmp_path / 'class1'
    class2_dir = tmp_path / 'class2'
    os.makedirs(class1_dir)
    os.makedirs(class2_dir)
    
    # Create sample images
    for i in range(10):
        (class1_dir / f'img1_{i}.jpg').touch()
        (class2_dir / f'img2_{i}.jpg').touch()
    
    return tmp_path

def test_split_dataset(sample_dataset, tmp_path):
    """Test dataset splitting functionality."""
    output_path = tmp_path / 'output'
    os.makedirs(output_path)
    
    # Split dataset
    train_path, val_path, test_path = split_dataset(
        sample_dataset,
        output_path,
        test_size=0.3
    )
    
    # Check if directories are created
    assert os.path.exists(train_path)
    assert os.path.exists(val_path)
    assert os.path.exists(test_path)
    
    # Count files in each split
    train_files = sum(1 for _ in os.listdir(train_path))
    val_files = sum(1 for _ in os.listdir(val_path))
    test_files = sum(1 for _ in os.listdir(test_path))
    
    # Test split proportions (approximately)
    total_files = train_files + val_files + test_files
    assert total_files == 20  # Total number of sample files
    
    # Check approximate split ratios (allowing for small variations due to rounding)
    assert 13 <= train_files <= 15  # ~70% of data
    assert 2 <= val_files <= 4      # ~15% of data
    assert 2 <= test_files <= 4     # ~15% of data