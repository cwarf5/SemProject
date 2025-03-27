import cv2
import argparse
import numpy as np
from datetime import datetime
import os
from data import WildfireDataLoader
from model import WildfireDetectionModel
# Add imports for Pi camera
import io
try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    PI_CAMERA_AVAILABLE = True
except ImportError:
    PI_CAMERA_AVAILABLE = False
    print("Warning: picamera module not found. Pi camera will not be available.")

class WebcamFireDetector:
    def __init__(self, model_path, img_size=(224, 224), threshold=0.5, use_pi_camera=False):
        """
        Initialize the webcam-based fire detector.
        
        Args:
            model_path (str): Path to the trained model
            img_size (tuple): Input image size (height, width)
            threshold (float): Classification threshold
            use_pi_camera (bool): Whether to use Pi camera instead of webcam
        """
        self.img_size = img_size
        self.threshold = threshold
        self.model = WildfireDetectionModel(img_size=img_size)
        self.model.load_model(model_path)
        self.data_loader = WildfireDataLoader(data_dir="", img_size=img_size)
        self.use_pi_camera = use_pi_camera
        
        # Initialize camera
        self.cap = None
        self.pi_camera = None
        
        # Create directory for saving detections
        self.detection_dir = "detections"
        os.makedirs(self.detection_dir, exist_ok=True)
    
    def preprocess_frame(self, frame):
        """
        Preprocess a video frame for prediction.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Preprocessed frame
        """
        # Resize frame
        resized = cv2.resize(frame, self.img_size)
        # Convert to RGB (from BGR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize
        normalized = rgb.astype('float32') / 255.0
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        return batched
    
    def save_detection(self, frame):
        """
        Save frame when fire is detected.
        
        Args:
            frame (numpy.ndarray): Frame to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.detection_dir, f"fire_detection_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Detection saved: {filename}")
    
    def start_detection(self, camera_id=0, display_size=(640, 480), pi_resolution=(640, 480), pi_framerate=32):
        """
        Start real-time fire detection from camera feed.
        
        Args:
            camera_id (int): Camera device ID (for webcam)
            display_size (tuple): Size of the display window
            pi_resolution (tuple): Pi camera resolution
            pi_framerate (int): Pi camera framerate
        """
        try:
            if self.use_pi_camera:
                if not PI_CAMERA_AVAILABLE:
                    raise ImportError("PiCamera module is not available")
                
                # Initialize Pi camera
                self.pi_camera = PiCamera()
                self.pi_camera.resolution = pi_resolution
                self.pi_camera.framerate = pi_framerate
                raw_capture = PiRGBArray(self.pi_camera, size=pi_resolution)
                
                # Allow camera to warm up
                print("Starting Pi camera. Please wait...")
                import time
                time.sleep(2)
                print("Starting real-time fire detection. Press 'q' to quit.")
                
                # Capture frames from Pi camera
                for frame in self.pi_camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
                    # Get frame from Pi camera
                    frame_data = frame.array
                    
                    # Resize frame for display
                    display_frame = cv2.resize(frame_data, display_size)
                    
                    # Preprocess frame for prediction
                    preprocessed = self.preprocess_frame(frame_data)
                    
                    # Make prediction
                    confidence = self.model.predict(preprocessed)
                    prediction = "FIRE DETECTED!" if confidence >= self.threshold else "No Fire"
                    
                    # Add prediction text to frame
                    color = (0, 0, 255) if confidence >= self.threshold else (0, 255, 0)
                    text = f"{prediction} ({confidence:.2f})"
                    cv2.putText(display_frame, text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Save frame if fire is detected
                    if confidence >= self.threshold:
                        self.save_detection(frame_data)
                    
                    # Display frame
                    cv2.imshow('Wildfire Detection', display_frame)
                    
                    # Clear the stream for next frame
                    raw_capture.truncate(0)
                    
                    # Check for 'q' key to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                # Use regular webcam
                self.cap = cv2.VideoCapture(camera_id)
                if not self.cap.isOpened():
                    raise ValueError("Could not open webcam")
                
                print("Starting real-time fire detection. Press 'q' to quit.")
                
                while True:
                    # Read frame
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Error reading frame")
                        break
                    
                    # Resize frame for display
                    display_frame = cv2.resize(frame, display_size)
                    
                    # Preprocess frame for prediction
                    preprocessed = self.preprocess_frame(frame)
                    
                    # Make prediction
                    confidence = self.model.predict(preprocessed)
                    prediction = "FIRE DETECTED!" if confidence >= self.threshold else "No Fire"
                    
                    # Add prediction text to frame
                    color = (0, 0, 255) if confidence >= self.threshold else (0, 255, 0)
                    text = f"{prediction} ({confidence:.2f})"
                    cv2.putText(display_frame, text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Save frame if fire is detected
                    if confidence >= self.threshold:
                        self.save_detection(frame)
                    
                    # Display frame
                    cv2.imshow('Wildfire Detection', display_frame)
                    
                    # Check for 'q' key to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
        except Exception as e:
            print(f"Error during camera detection: {str(e)}")
        
        finally:
            if self.cap is not None:
                self.cap.release()
            if self.pi_camera is not None:
                self.pi_camera.close()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Real-time wildfire detection using camera')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--camera_id', type=int, default=0,
                      help='Camera device ID (for webcam)')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Detection threshold')
    parser.add_argument('--img_size', type=int, default=224,
                      help='Image size for model input')
    parser.add_argument('--display_width', type=int, default=640,
                      help='Display window width')
    parser.add_argument('--display_height', type=int, default=480,
                      help='Display window height')
    parser.add_argument('--use_pi_camera', action='store_true',
                      help='Use Raspberry Pi camera module instead of webcam')
    parser.add_argument('--pi_width', type=int, default=640,
                      help='Pi camera resolution width')
    parser.add_argument('--pi_height', type=int, default=480,
                      help='Pi camera resolution height')
    parser.add_argument('--pi_framerate', type=int, default=32,
                      help='Pi camera framerate')
    
    args = parser.parse_args()
    
    detector = WebcamFireDetector(
        model_path=args.model_path,
        img_size=(args.img_size, args.img_size),
        threshold=args.threshold,
        use_pi_camera=args.use_pi_camera
    )
    
    detector.start_detection(
        camera_id=args.camera_id,
        display_size=(args.display_width, args.display_height),
        pi_resolution=(args.pi_width, args.pi_height),
        pi_framerate=args.pi_framerate
    )

if __name__ == '__main__':
    main()