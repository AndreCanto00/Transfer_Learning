import os
import shutil
import pytest
from src.utils.file_operations import listdir, organize_files

@pytest.fixture
def temp_directory(tmp_path):
    """Create a temporary directory with some test files."""
    # Create test files
    test_files = [
        '.hidden_file',
        'test123_001.jpg',
        'test123_002.jpg',
        'test456_001.jpg'
    ]
    
    for file_name in test_files:
        file_path = tmp_path / file_name
        file_path.touch()
    
    return tmp_path

def test_listdir(temp_directory):
    """Test listdir function excludes hidden files."""
    files = list(listdir(temp_directory))
    
    assert len(files) == 3  # Should exclude .hidden_file
    assert '.hidden_file' not in files
    assert 'test123_001.jpg' in files
    assert 'test123_002.jpg' in files
    assert 'test456_001.jpg' in files

def test_organize_files(temp_directory):
    """Test organize_files function creates correct directory structure."""
    # Create a destination directory
    dest_dir = temp_directory / 'organized'
    os.makedirs(dest_dir)
    
    # Organize files
    organize_files(temp_directory, dest_dir)
    
    # Check if files are organized correctly
    assert os.path.exists(dest_dir / 'test123_0')
    assert os.path.exists(dest_dir / 'test456_0')
    assert os.path.exists(dest_dir / 'test123_0' / 'test123_001.jpg')
    assert os.path.exists(dest_dir / 'test123_0' / 'test123_002.jpg')
    assert os.path.exists(dest_dir / 'test456_0' / 'test456_001.jpg')