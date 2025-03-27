import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import random

class WildfireVisualizer:
    def __init__(self, data_dir=None, model=None, history=None):
        """
        Initialize the visualizer.
        
        Args:
            data_dir (str): Root directory containing the dataset
            model: Trained model instance
            history: Training history object
        """
        self.data_dir = data_dir
        self.model = model
        self.history = history
        
    def plot_dataset_distribution(self, save_path=None):
        """
        Plot the distribution of images across classes.
        """
        if not self.data_dir:
            raise ValueError("Data directory not provided")
            
        # Define directory paths
        train_fire_dir = os.path.join(self.data_dir, 'train', 'fire')
        train_no_fire_dir = os.path.join(self.data_dir, 'train', 'no_fire')
        val_fire_dir = os.path.join(self.data_dir, 'validation', 'fire')
        val_no_fire_dir = os.path.join(self.data_dir, 'validation', 'no_fire')
        
        # Initialize counts
        train_fire_count = 0
        train_no_fire_count = 0
        val_fire_count = 0
        val_no_fire_count = 0
        
        # Count training images if directories exist
        if os.path.exists(train_fire_dir):
            train_fire_count = len([f for f in os.listdir(train_fire_dir)
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if os.path.exists(train_no_fire_dir):
            train_no_fire_count = len([f for f in os.listdir(train_no_fire_dir)
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Count validation images if directories exist
        if os.path.exists(val_fire_dir):
            val_fire_count = len([f for f in os.listdir(val_fire_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if os.path.exists(val_no_fire_dir):
            val_no_fire_count = len([f for f in os.listdir(val_no_fire_dir)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Print statistics
        print("\nDataset Statistics:")
        print(f"Training:")
        print(f"  - Fire images: {train_fire_count}")
        print(f"  - No fire images: {train_no_fire_count}")
        print(f"Validation:")
        print(f"  - Fire images: {val_fire_count}")
        print(f"  - No fire images: {val_no_fire_count}")
        
        # Create grouped bar plot
        labels = ['Fire', 'No Fire']
        train_counts = [train_fire_count, train_no_fire_count]
        val_counts = [val_fire_count, val_no_fire_count]
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, train_counts, width, label='Training')
        ax.bar(x + width/2, val_counts, width, label='Validation')
        
        ax.set_ylabel('Number of Images')
        ax.set_title('Dataset Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        # Add value labels on top of bars
        for i, count in enumerate(train_counts):
            ax.text(i - width/2, count, str(count), ha='center', va='bottom')
        for i, count in enumerate(val_counts):
            ax.text(i + width/2, count, str(count), ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        
    def plot_sample_images(self, num_samples=5, save_path=None):
        """
        Plot sample images from each class.
        """
        if not self.data_dir:
            raise ValueError("Data directory not provided")
            
        # Define directory paths
        train_fire_dir = os.path.join(self.data_dir, 'train', 'fire')
        train_no_fire_dir = os.path.join(self.data_dir, 'train', 'no_fire')
        
        # Check if directories exist
        if not os.path.exists(train_fire_dir) or not os.path.exists(train_no_fire_dir):
            print(f"Warning: Training directories not found in {self.data_dir}/train/")
            return
            
        # Get list of valid image files
        fire_files = [f for f in os.listdir(train_fire_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        no_fire_files = [f for f in os.listdir(train_no_fire_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Adjust num_samples if there aren't enough images
        num_samples = min(num_samples, len(fire_files), len(no_fire_files))
        if num_samples == 0:
            print("Warning: No valid images found in training directories")
            return
            
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        
        # Plot fire images from training set
        for i, file in enumerate(random.sample(fire_files, num_samples)):
            img = Image.open(os.path.join(train_fire_dir, file))
            axes[0, i].imshow(img)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Fire', pad=20)
        
        # Plot no-fire images from training set
        for i, file in enumerate(random.sample(no_fire_files, num_samples)):
            img = Image.open(os.path.join(train_no_fire_dir, file))
            axes[1, i].imshow(img)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('No Fire', pad=20)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        
    def plot_training_history(self, save_path=None):
        """
        Plot training history metrics.
        """
        if not self.history:
            raise ValueError("Training history not provided")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Plot confusion matrix for model predictions.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_prediction_confidence(self, image_path, save_path=None):
        """
        Plot model's prediction confidence for a single image.
        """
        if not self.model:
            raise ValueError("Model not provided")
            
        # Load and preprocess image
        img = Image.open(image_path)
        img_array = np.array(img.resize((224, 224))) / 255.0
        prediction = self.model.predict(np.expand_dims(img_array, axis=0))[0][0]
        
        plt.figure(figsize=(12, 4))
        
        # Plot image
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Input Image')
        
        # Plot confidence
        plt.subplot(1, 2, 2)
        bars = plt.bar(['No Fire', 'Fire'], [1-prediction, prediction])
        plt.title('Prediction Confidence')
        plt.ylim(0, 1)
        
        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height*100:.1f}%',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def generate_evaluation_report(self, y_true, y_pred, save_path=None):
        """
        Generate and optionally save a complete evaluation report.
        """
        report = classification_report(y_true, y_pred, target_names=['No Fire', 'Fire'])
        print("\nClassification Report:")
        print(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report) 