import os
import argparse
import numpy as np
from data import WildfireDataLoader
from model import WildfireDetectionModel

class WildfirePredictor:
    def __init__(self, model_path, img_size=(224, 224)):
        """
        Initialize the wildfire predictor.
        
        Args:
            model_path (str): Path to the saved model
            img_size (tuple): Input image size (height, width)
        """
        self.img_size = img_size
        self.model = WildfireDetectionModel(img_size=img_size)
        self.model.load_model(model_path)
        self.data_loader = WildfireDataLoader(data_dir="", img_size=img_size)
    
    def predict_image(self, image_path, threshold=0.5):
        """
        Make a prediction on a single image.
        
        Args:
            image_path (str): Path to the image file
            threshold (float): Classification threshold
            
        Returns:
            tuple: (prediction label, confidence score)
        """
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Preprocess the image
        preprocessed_image = self.data_loader.preprocess_single_image(image_path)
        
        # Make prediction
        confidence = self.model.predict(preprocessed_image)
        prediction = "fire" if confidence >= threshold else "no fire"
        
        return prediction, confidence

def main():
    parser = argparse.ArgumentParser(description='Predict wildfire in images')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to the image file to predict')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Classification threshold')
    parser.add_argument('--img_size', type=int, default=224,
                      help='Image size (width and height)')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = WildfirePredictor(
            model_path=args.model_path,
            img_size=(args.img_size, args.img_size)
        )
        
        # Make prediction
        prediction, confidence = predictor.predict_image(
            args.image_path,
            threshold=args.threshold
        )
        
        # Print results
        print(f"\nImage: {args.image_path}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.4f}")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == '__main__':
    main() 