import os
import sys
from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("yolo_fire_detection.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def train_yolo_model(data_yaml_path, epochs=50, imgsz=640, batch=8):
    """
    Train a YOLOv8 model for fire detection.
    
    Args:
        data_yaml_path: Path to YAML file with dataset configuration
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
    
    Returns:
        Path to the best trained model
    """
    try:
        # Load a model
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

        # Train the model
        logger.info(f"Starting YOLOv8 training for {epochs} epochs...")
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name='fire_detection'
        )
        
        # Get the path to the best model
        best_model_path = f"runs/detect/fire_detection/weights/best.pt"
        if os.path.exists(best_model_path):
            logger.info(f"Training completed. Best model saved to {best_model_path}")
            return best_model_path
        else:
            logger.error(f"Model training completed but best model not found at {best_model_path}")
            return None
            
    except Exception as e:
        logger.error(f"Error during YOLOv8 training: {e}")
        return None

def create_data_yaml(root_folder, output_path="data.yaml"):
    """
    Create a YAML file for YOLOv8 training from the existing dataset structure.
    
    Args:
        root_folder: Root folder containing train and validation directories
        output_path: Path to save the YAML file
    
    Returns:
        Path to the created YAML file
    """
    try:
        train_path = os.path.join(root_folder, 'train')
        val_path = os.path.join(root_folder, 'validation')
        
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            logger.error(f"Train or validation folders not found in {root_folder}")
            return None
            
        # Define class names
        class_names = ['fire', 'no_fire']
        
        # Create YAML content
        yaml_content = f"""
# Dataset path
path: {root_folder}
train: train
val: validation

# Classes
names:
  0: fire
  1: no_fire
"""
        
        # Write YAML file
        with open(output_path, 'w') as f:
            f.write(yaml_content)
            
        logger.info(f"Created YOLOv8 dataset configuration at {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating data YAML file: {e}")
        return None

def convert_dataset_to_yolo_format(root_folder, output_folder):
    """
    Convert the existing dataset to YOLOv8 format.
    YOLOv8 expects images in a folder and labels in a separate folder.
    Each label is a text file with the same name as the image,
    containing bounding box coordinates.
    
    Args:
        root_folder: Root folder containing train and validation directories
        output_folder: Folder to save the converted dataset
    
    Returns:
        Path to the output folder
    """
    try:
        for split in ['train', 'validation']:
            for class_name in ['fire', 'no_fire']:
                # Source folder
                src_folder = os.path.join(root_folder, split, class_name)
                
                # Create output folders
                images_folder = os.path.join(output_folder, split, 'images')
                labels_folder = os.path.join(output_folder, split, 'labels')
                os.makedirs(images_folder, exist_ok=True)
                os.makedirs(labels_folder, exist_ok=True)
                
                # Process each image in the folder
                if os.path.exists(src_folder):
                    for img_file in os.listdir(src_folder):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            # Copy image to images folder
                            src_path = os.path.join(src_folder, img_file)
                            dst_path = os.path.join(images_folder, img_file)
                            img = cv2.imread(src_path)
                            if img is not None:
                                cv2.imwrite(dst_path, img)
                                
                                # Create label file (assuming the whole image contains the class)
                                label_file = os.path.splitext(img_file)[0] + '.txt'
                                label_path = os.path.join(labels_folder, label_file)
                                
                                # For fire class, create a bounding box covering the image
                                if class_name == 'fire':
                                    h, w = img.shape[:2]
                                    # Format: class_id center_x center_y width height (normalized)
                                    # For fire class, use class_id 0
                                    with open(label_path, 'w') as f:
                                        f.write(f"0 0.5 0.5 1.0 1.0\n")  # Full image bbox
                                else:
                                    # For no_fire class, create an empty label file or skip
                                    # depending on your needs
                                    # If you want to include no_fire as a class, use class_id 1
                                    with open(label_path, 'w') as f:
                                        f.write(f"1 0.5 0.5 1.0 1.0\n")  # Full image bbox
        
        logger.info(f"Dataset converted to YOLOv8 format at {output_folder}")
        return output_folder
        
    except Exception as e:
        logger.error(f"Error converting dataset to YOLO format: {e}")
        return None

def test_yolo_model(model_path, test_folder, output_folder="results"):
    """
    Test a trained YOLOv8 model on test images.
    
    Args:
        model_path: Path to the trained model
        test_folder: Folder containing test images
        output_folder: Folder to save results
    """
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            return
            
        # Load the model
        model = YOLO(model_path)
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Process test images
        for class_folder in ['fire', 'no_fire']:
            class_path = os.path.join(test_folder, class_folder)
            if os.path.exists(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, img_file)
                        
                        # Run prediction
                        results = model.predict(img_path, save=True, project=output_folder)
                        
                        # Log prediction results
                        for r in results:
                            boxes = r.boxes
                            if len(boxes) > 0:
                                logger.info(f"Detected {len(boxes)} objects in {img_file}")
                                for box in boxes:
                                    cls = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    logger.info(f"  Class: {cls}, Confidence: {conf:.4f}")
        
        logger.info(f"Test results saved to {output_folder}")
        
    except Exception as e:
        logger.error(f"Error testing YOLOv8 model: {e}")

def main():
    """Main function to run the YOLOv8 training and testing pipeline."""
    # Define paths
    root_folder = os.path.abspath("../firedetect/data")
    yolo_dataset_folder = os.path.abspath("../firedetect/data_yolo")
    data_yaml_path = os.path.abspath("../firedetect/data.yaml")
    test_folder = os.path.abspath("../firedetect/test")
    output_folder = os.path.abspath("../firedetect/results")
    
    # Convert dataset to YOLOv8 format
    logger.info("Converting dataset to YOLOv8 format...")
    convert_dataset_to_yolo_format(root_folder, yolo_dataset_folder)
    
    # Create data YAML
    logger.info("Creating data YAML file...")
    create_data_yaml(yolo_dataset_folder, data_yaml_path)
    
    # Train the model
    logger.info("Training YOLOv8 model...")
    model_path = train_yolo_model(data_yaml_path, epochs=30, imgsz=640, batch=8)
    
    if model_path and os.path.exists(model_path):
        # Test the model
        logger.info("Testing YOLOv8 model...")
        test_yolo_model(model_path, test_folder, output_folder)
        
        # Copy the model to models folder
        models_folder = os.path.abspath("../firedetect/models")
        os.makedirs(models_folder, exist_ok=True)
        yolo_model_path = os.path.join(models_folder, "yolov8_fire_detection.pt")
        import shutil
        shutil.copy(model_path, yolo_model_path)
        logger.info(f"Model copied to {yolo_model_path}")
    else:
        logger.error("Training failed or model not found")

if __name__ == "__main__":
    main()