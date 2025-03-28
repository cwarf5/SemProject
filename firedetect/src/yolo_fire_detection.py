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

def train_yolo_model(data_yaml_path, epochs=10, imgsz=640, batch=8):
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
        
        # Try different possible paths for the best model
        possible_paths = [
            "runs/detect/fire_detection/weights/best.pt",
            f"runs/detect/fire_detection{epochs}/weights/best.pt",
            f"runs/detect/fire_detection3/weights/best.pt"  # Specific path from your error
        ]
        
        # Look for the best model in all possible locations
        for model_path in possible_paths:
            if os.path.exists(model_path):
                logger.info(f"Training completed. Best model found at {model_path}")
                return model_path
                
        # If we get here, check if any "runs/detect" directory exists with a weights subfolder
        if os.path.exists("runs/detect"):
            for run_dir in os.listdir("runs/detect"):
                weights_dir = os.path.join("runs/detect", run_dir, "weights")
                if os.path.exists(weights_dir):
                    for file in os.listdir(weights_dir):
                        if file == "best.pt":
                            model_path = os.path.join("runs/detect", run_dir, "weights", file)
                            logger.info(f"Training completed. Best model found at {model_path}")
                            return model_path
        
        logger.error(f"Model training completed but best model not found in any expected location")
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
        # Make sure the paths exist
        train_images = os.path.join(root_folder, 'train', 'images')
        train_labels = os.path.join(root_folder, 'train', 'labels')
        val_images = os.path.join(root_folder, 'validation', 'images')
        val_labels = os.path.join(root_folder, 'validation', 'labels')
        
        for path in [train_images, train_labels, val_images, val_labels]:
            if not os.path.exists(path):
                logger.error(f"Required path not found: {path}")
                return None
            
        # Define class names
        class_names = ['fire', 'no_fire']
        
        # Create YAML content - properly format paths for YOLOv8
        yaml_content = f"""
# Dataset path
path: {root_folder}
train: train/images
val: validation/images
test: # No test split

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
    YOLOv8 expects images in an 'images' folder and labels in a 'labels' folder.
    Each label is a text file with the same name as the image,
    containing bounding box coordinates in normalized format.
    
    Args:
        root_folder: Root folder containing train and validation directories
        output_folder: Folder to save the converted dataset
    
    Returns:
        Path to the output folder
    """
    try:
        for split in ['train', 'validation']:
            # Create output folders for this split
            images_folder = os.path.join(output_folder, split, 'images')
            labels_folder = os.path.join(output_folder, split, 'labels')
            os.makedirs(images_folder, exist_ok=True)
            os.makedirs(labels_folder, exist_ok=True)
            
            # Process fire class images
            fire_folder = os.path.join(root_folder, split, 'fire')
            if os.path.exists(fire_folder):
                for img_file in os.listdir(fire_folder):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Copy image to images folder
                        src_path = os.path.join(fire_folder, img_file)
                        dst_path = os.path.join(images_folder, img_file)
                        
                        try:
                            img = cv2.imread(src_path)
                            if img is not None:
                                cv2.imwrite(dst_path, img)
                                logger.info(f"Copied fire image: {dst_path}")
                                
                                # Create label file for fire class (class 0)
                                label_file = os.path.splitext(img_file)[0] + '.txt'
                                label_path = os.path.join(labels_folder, label_file)
                                
                                # Format: class_id center_x center_y width height (normalized)
                                with open(label_path, 'w') as f:
                                    f.write("0 0.5 0.5 1.0 1.0\n")  # Full image bbox for class 0 (fire)
                        except Exception as e:
                            logger.error(f"Error processing fire image {src_path}: {e}")
            
            # Process no_fire class images
            no_fire_folder = os.path.join(root_folder, split, 'no_fire')
            if os.path.exists(no_fire_folder):
                for img_file in os.listdir(no_fire_folder):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Copy image to images folder
                        src_path = os.path.join(no_fire_folder, img_file)
                        dst_path = os.path.join(images_folder, img_file)
                        
                        try:
                            img = cv2.imread(src_path)
                            if img is not None:
                                cv2.imwrite(dst_path, img)
                                logger.info(f"Copied no_fire image: {dst_path}")
                                
                                # Create label file for no_fire class (class 1)
                                label_file = os.path.splitext(img_file)[0] + '.txt'
                                label_path = os.path.join(labels_folder, label_file)
                                
                                # Format: class_id center_x center_y width height (normalized)
                                with open(label_path, 'w') as f:
                                    f.write("1 0.5 0.5 1.0 1.0\n")  # Full image bbox for class 1 (no_fire)
                        except Exception as e:
                            logger.error(f"Error processing no_fire image {src_path}: {e}")
        
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
    # Define paths - fix the path structure to avoid nested directories
    root_folder = os.path.abspath("../data")
    yolo_dataset_folder = os.path.abspath("../data_yolo")
    data_yaml_path = os.path.abspath("../data.yaml")
    test_folder = os.path.abspath("../test")
    output_folder = os.path.abspath("../results")
    
    # Convert dataset to YOLOv8 format
    logger.info("Converting dataset to YOLOv8 format...")
    convert_dataset_to_yolo_format(root_folder, yolo_dataset_folder)
    
    # Create data YAML
    logger.info("Creating data YAML file...")
    create_data_yaml(yolo_dataset_folder, data_yaml_path)
    
    # Train the model
    logger.info("Training YOLOv8 model...")
    model_path = train_yolo_model(data_yaml_path, epochs=30, imgsz=640, batch=8)
    
    # If model_path is still None, try to find the model directly from the recent runs
    if model_path is None:
        # Check the specific path from the error message
        specific_path = os.path.abspath("/home/pi/finalproject/SemProject/runs/detect/fire_detection3/weights/best.pt")
        if os.path.exists(specific_path):
            logger.info(f"Found model at path from error message: {specific_path}")
            model_path = specific_path
    
    if model_path and os.path.exists(model_path):
        # Test the model
        logger.info("Testing YOLOv8 model...")
        test_yolo_model(model_path, test_folder, output_folder)
        
        # Copy the model to models folder
        models_folder = os.path.abspath("../models")
        os.makedirs(models_folder, exist_ok=True)
        yolo_model_path = os.path.join(models_folder, "yolov8_fire_detection.pt")
        import shutil
        shutil.copy(model_path, yolo_model_path)
        logger.info(f"Model copied to {yolo_model_path}")
    else:
        logger.error("Training failed or model not found")

if __name__ == "__main__":
    main()