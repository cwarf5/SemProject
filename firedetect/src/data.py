import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class WildfireDataLoader:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize the data loader for wildfire detection.
        
        Args:
            data_dir (str): Root directory containing 'train' and 'validation' subdirectories
            img_size (tuple): Target size for images (height, width)
            batch_size (int): Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_dir = os.path.join(data_dir, 'train')
        self.val_dir = os.path.join(data_dir, 'validation')
        
        # Data augmentation for training
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        self.val_datagen = ImageDataGenerator(
            rescale=1./255
        )

    def load_data_generators(self):
        """
        Create train and validation data generators.
        
        Returns:
            tuple: (train_generator, validation_generator)
        """
        train_generator = self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['no_fire', 'fire']
        )
        
        validation_generator = self.val_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['no_fire', 'fire']
        )
        
        return train_generator, validation_generator

    def preprocess_single_image(self, image_path):
        """
        Preprocess a single image for inference.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image ready for model input
        """
        img = Image.open(image_path)
        img = img.resize(self.img_size)
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def validate_data_directory(self):
        """
        Validate that the data directory has the correct structure and contains images.
        
        Returns:
            bool: True if valid, False otherwise
        """
        # Check if main directories exist
        if not os.path.exists(self.train_dir) or not os.path.exists(self.val_dir):
            print(f"Error: Missing required directories in {self.data_dir}")
            return False
        
        # Check training directories
        train_fire_dir = os.path.join(self.train_dir, 'fire')
        train_no_fire_dir = os.path.join(self.train_dir, 'no_fire')
        
        # Check validation directories
        val_fire_dir = os.path.join(self.val_dir, 'fire')
        val_no_fire_dir = os.path.join(self.val_dir, 'no_fire')
        
        # Check if all subdirectories exist
        for dir_path in [train_fire_dir, train_no_fire_dir, val_fire_dir, val_no_fire_dir]:
            if not os.path.exists(dir_path):
                print(f"Error: Missing directory {dir_path}")
                return False
        
        # Count images in training directories
        train_fire_images = len([f for f in os.listdir(train_fire_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        train_no_fire_images = len([f for f in os.listdir(train_no_fire_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Count images in validation directories
        val_fire_images = len([f for f in os.listdir(val_fire_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        val_no_fire_images = len([f for f in os.listdir(val_no_fire_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Check if any directory is empty
        if train_fire_images == 0 or train_no_fire_images == 0 or \
           val_fire_images == 0 or val_no_fire_images == 0:
            print(f"Error: Empty directories found")
            print(f"Training fire images: {train_fire_images}")
            print(f"Training no-fire images: {train_no_fire_images}")
            print(f"Validation fire images: {val_fire_images}")
            print(f"Validation no-fire images: {val_no_fire_images}")
            return False
        
        print("\nDataset Statistics:")
        print(f"Training:")
        print(f"  - Fire images: {train_fire_images}")
        print(f"  - No fire images: {train_no_fire_images}")
        print(f"Validation:")
        print(f"  - Fire images: {val_fire_images}")
        print(f"  - No fire images: {val_no_fire_images}")
        
        return True 