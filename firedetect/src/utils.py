"""
Utility functions for the wildfire detection system.

This module contains common utility functions used across the codebase
to ensure consistency and avoid code duplication.
"""

import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def count_images_in_directory(directory_path):
    """
    Count the number of image files in a directory.
    
    Args:
        directory_path (str): Path to the directory containing images
        
    Returns:
        int: Number of image files in the directory
    """
    if not os.path.exists(directory_path):
        logger.warning(f"Directory not found: {directory_path}")
        return 0
    
    return len([f for f in os.listdir(directory_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

def get_dataset_statistics(data_dir):
    """
    Get statistics about the dataset.
    
    Args:
        data_dir (str): Root directory containing the dataset
        
    Returns:
        dict: Dictionary containing dataset statistics
    """
    # Define directory paths
    train_fire_dir = os.path.join(data_dir, 'train', 'fire')
    train_no_fire_dir = os.path.join(data_dir, 'train', 'no_fire')
    val_fire_dir = os.path.join(data_dir, 'validation', 'fire')
    val_no_fire_dir = os.path.join(data_dir, 'validation', 'no_fire')
    
    # Count images
    train_fire_count = count_images_in_directory(train_fire_dir)
    train_no_fire_count = count_images_in_directory(train_no_fire_dir)
    val_fire_count = count_images_in_directory(val_fire_dir)
    val_no_fire_count = count_images_in_directory(val_no_fire_dir)
    
    return {
        'train_fire': train_fire_count,
        'train_no_fire': train_no_fire_count,
        'val_fire': val_fire_count,
        'val_no_fire': val_no_fire_count,
        'train_total': train_fire_count + train_no_fire_count,
        'val_total': val_fire_count + val_no_fire_count,
        'total': train_fire_count + train_no_fire_count + val_fire_count + val_no_fire_count
    }

def validate_img_size(img_size):
    """
    Validate and standardize image size parameter.
    
    Args:
        img_size: Image size parameter, can be int or tuple
        
    Returns:
        tuple: Standardized (height, width) tuple
    """
    if isinstance(img_size, int):
        return (img_size, img_size)
    elif isinstance(img_size, tuple) and len(img_size) == 2:
        return img_size
    else:
        logger.warning(f"Invalid image size: {img_size}. Using default (224, 224).")
        return (224, 224)