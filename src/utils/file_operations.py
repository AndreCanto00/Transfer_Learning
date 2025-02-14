import os
import shutil
from os.path import isfile, join, isdir

def listdir(path):
    """
    List directory contents, excluding hidden files.
    
    Args:
        path (str): Path to directory
    
    Yields:
        str: Name of each non-hidden file/directory
    """
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def organize_files(source_path, destination_path):
    """
    Organize files into subdirectories based on filename prefix.
    
    Args:
        source_path (str): Source directory containing files
        destination_path (str): Destination directory for organized files
    """
    for file_name in os.listdir(source_path):
        source_file_path = os.path.join(source_path, file_name)
        if os.path.isfile(source_file_path):
            prefix = file_name[:9]
            destination_folder = os.path.join(destination_path, prefix)
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
            destination_file_path = os.path.join(destination_folder, file_name)
            shutil.move(source_file_path, destination_file_path)