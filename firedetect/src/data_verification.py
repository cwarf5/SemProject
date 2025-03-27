import os
import shutil
import cv2
import numpy as np
from PIL import Image
import argparse

class DatasetVerifier:
    def __init__(self, data_dir):
        """
        Initialize the dataset verifier.
        
        Args:
            data_dir (str): Root directory containing 'fire' and 'no_fire' subdirectories
        """
        self.data_dir = data_dir
        self.fire_dir = os.path.join(data_dir, 'fire')
        self.no_fire_dir = os.path.join(data_dir, 'no_fire')
        self.review_dir = os.path.join(data_dir, 'review')  # For uncertain cases
        
        # Create review directory if it doesn't exist
        os.makedirs(self.review_dir, exist_ok=True)
    
    def verify_images(self):
        """
        Interactive tool to verify and clean the dataset.
        Shows each image and allows user to:
        - Confirm current classification
        - Move to opposite class
        - Move to review folder
        - Delete the image
        """
        for class_name in ['fire', 'no_fire']:
            class_dir = os.path.join(self.data_dir, class_name)
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"\nVerifying {class_name} images...")
            print(f"Total images: {len(images)}")
            
            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                
                # Display image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Error loading image: {img_path}")
                    continue
                
                img = cv2.resize(img, (800, 600))
                cv2.imshow('Image Verification', img)
                
                # Instructions
                print(f"\nCurrent image: {img_name}")
                print("Current class:", class_name)
                print("\nOptions:")
                print("'k' - Keep in current class")
                print("'m' - Move to opposite class")
                print("'r' - Move to review folder")
                print("'d' - Delete image")
                print("'q' - Quit verification")
                
                # Wait for keypress
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                
                elif key == ord('k'):
                    print(f"Keeping {img_name} in {class_name}")
                    continue
                
                elif key == ord('m'):
                    # Move to opposite class
                    target_class = 'no_fire' if class_name == 'fire' else 'fire'
                    target_path = os.path.join(self.data_dir, target_class, img_name)
                    shutil.move(img_path, target_path)
                    print(f"Moved {img_name} to {target_class}")
                
                elif key == ord('r'):
                    # Move to review folder
                    review_path = os.path.join(self.review_dir, img_name)
                    shutil.move(img_path, review_path)
                    print(f"Moved {img_name} to review folder")
                
                elif key == ord('d'):
                    # Delete image
                    os.remove(img_path)
                    print(f"Deleted {img_name}")
            
        cv2.destroyAllWindows()
    
    def generate_statistics(self):
        """
        Generate and print statistics about the dataset.
        """
        stats = {}
        for class_name in ['fire', 'no_fire', 'review']:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                images = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                stats[class_name] = len(images)
        
        print("\nDataset Statistics:")
        print(f"Fire images: {stats.get('fire', 0)}")
        print(f"No fire images: {stats.get('no_fire', 0)}")
        print(f"Images in review: {stats.get('review', 0)}")
        
        return stats

def main():
    parser = argparse.ArgumentParser(description='Verify and clean wildfire detection dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing fire and no_fire subdirectories')
    
    args = parser.parse_args()
    
    verifier = DatasetVerifier(args.data_dir)
    
    # Print initial statistics
    print("Initial dataset statistics:")
    verifier.generate_statistics()
    
    # Run verification process
    verifier.verify_images()
    
    # Print final statistics
    print("\nFinal dataset statistics after verification:")
    verifier.generate_statistics()

if __name__ == '__main__':
    main() 