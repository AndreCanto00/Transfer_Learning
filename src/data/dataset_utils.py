import os
import shutil
from sklearn.model_selection import train_test_split
from os.path import isfile, isdir, join
from ..utils.file_operations import listdir

def split_dataset(base_path, output_path, test_size=0.3, random_state=42):
    """
    Split dataset into train, validation and test sets.
    
    Args:
        base_path (str): Path to the original dataset
        output_path (str): Path where to save the split dataset
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random state for reproducibility
    """
    split_dataset_path = os.path.join(output_path, 'split_dataset')
    os.makedirs(split_dataset_path, exist_ok=True)
    
    # Create train, val, test directories
    train_path = os.path.join(split_dataset_path, 'train')
    val_path = os.path.join(split_dataset_path, 'val')
    test_path = os.path.join(split_dataset_path, 'test')
    
    for path in [train_path, val_path, test_path]:
        os.makedirs(path, exist_ok=True)
    
    # Get all classes
    classes = [class_name for class_name in listdir(base_path) 
              if isdir(os.path.join(base_path, class_name))]
    
    class_distribution = {'train': {}, 'val': {}, 'test': {}}
    
    for class_name in classes:
        class_folder = os.path.join(base_path, class_name)
        images = [img for img in listdir(class_folder) 
                 if isfile(join(class_folder, img))]
        
        if not images:
            continue  # Skip if no images in the class folder
        
        # Split into train and test-val
        train_images, test_val_images = train_test_split(
            images, test_size=test_size, random_state=random_state
        )
        
        # Split test-val into validation and test
        val_images, test_images = train_test_split(
            test_val_images, test_size=0.5, random_state=random_state
        )
        
        # Copy images to respective directories
        for img, target_path in [
            (train_images, train_path),
            (val_images, val_path),
            (test_images, test_path)
        ]:
            for image in img:
                shutil.copy(
                    os.path.join(class_folder, image),
                    os.path.join(target_path, image)
                )
        
        # Update class distribution
        class_distribution['train'][class_name] = len(train_images)
        class_distribution['val'][class_name] = len(val_images)
        class_distribution['test'][class_name] = len(test_images)
    
    print(f"Training set size: {len(train_images)}")
    print(f"Validation set size: {len(val_images)}")
    print(f"Test set size: {len(test_images)}")
    
    print("Class distribution:")
    for split, dist in class_distribution.items():
        print(f"{split}: {dist}")
    
    return train_path, val_path, test_path