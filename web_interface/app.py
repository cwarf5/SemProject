import io
import logging
import threading
import time
import cv2
import numpy as np
from flask import Flask, render_template, Response
import os
import configparser
import random
from tensorflow.keras.models import load_model
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Conditionally import picamera2 modules if available
try:
    from picamera2 import Picamera2, MappedArray
    from picamera2.encoders import JpegEncoder
    from picamera2.outputs import FileOutput
    from picamera2.devices import Hailo
    PI_AVAILABLE = True
except ImportError:
    PI_AVAILABLE = False
    logging.warning("Pi camera modules not available - running in simulation mode")

app = Flask(__name__)

# Global variable to store detections
detections = []

# Global fire model
fire_model = None

# Global streaming output instance
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()
    
    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

output = StreamingOutput()

# Load configuration
def load_config():
    """Load configuration from config.ini file or create default"""
    config = configparser.ConfigParser()
    
    # Default configuration
    config['DEFAULT'] = {
        'HailoModelPath': '/usr/share/hailo-models/yolov8s_h8l.hef',
        'LabelsPath': 'coco.txt',
        'FireModelPath': '../firedetect/models/best_model.h5',
        'ScoreThreshold': '0.5',
        'ImageWidth': '1280',
        'ImageHeight': '960',
        'FrameRate': '30'
    }
    
    # Try to load from config file if it exists
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
    if os.path.exists(config_path):
        config.read(config_path)
        logging.info(f"Loaded configuration from {config_path}")
    else:
        # Create default config file
        with open(config_path, 'w') as f:
            config.write(f)
        logging.info(f"Created default configuration file at {config_path}")
    
    return config['DEFAULT']

# Load configuration
config = load_config()
hailo_model_path = config['HailoModelPath']
labels_path = config['LabelsPath']
fire_model_path = config['FireModelPath']
score_threshold = float(config['ScoreThreshold'])
img_width = int(config['ImageWidth'])
img_height = int(config['ImageHeight'])
frame_rate = int(config['FrameRate'])

# Try to load the fire detection model directly
try:
    fire_model_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), fire_model_path))
    if os.path.exists(fire_model_path):
        # Use compile=False to avoid compatibility issues
        fire_model = load_model(fire_model_path, compile=False)
        logging.info(f"Loaded fire detection model from {fire_model_path}")
    else:
        logging.warning(f"Fire model not found at {fire_model_path}")
except Exception as e:
    logging.error(f"Error loading fire detection model: {e}")
    fire_model = None

# Load class names from the labels file
try:
    labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), labels_path)
    with open(labels_path, 'r', encoding="utf-8") as f:
        class_names = f.read().splitlines()
    logging.info(f"Loaded {len(class_names)} classes from {labels_path}")
except FileNotFoundError:
    class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "fire"]
    logging.warning(f"Labels file {labels_path} not found. Using default subset of classes.")

def preprocess_image(image):
    """Preprocess an image for the fire detection model."""
    # Convert OpenCV BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    image_resized = cv2.resize(image, (224, 224))
    
    # Normalize pixel values
    image_normalized = image_resized.astype('float32') / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch

def extract_detections(hailo_output, w, h, class_names, threshold=0.5):
    """Extract detections from the HailoRT-postprocess output."""
    results = []
    for class_id, dets in enumerate(hailo_output):
        for detection in dets:
            score = detection[4]
            if score >= threshold:
                y0, x0, y1, x1 = detection[:4]
                bbox = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
                results.append([class_names[class_id], bbox, score])
    return results

def detect_fire_in_frame(frame):
    """Detect fire in a frame using our fire detection model."""
    global fire_model
    
    if fire_model is None:
        return []
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(frame)
        
        # Get prediction
        prediction = fire_model.predict(processed_image, verbose=0)[0][0]
        
        # Check if fire detected based on threshold
        if prediction >= score_threshold:
            # Create a detection with the same format as other detections
            h, w = frame.shape[:2]
            x0, y0 = int(w * 0.25), int(h * 0.25)  # Approximated bounding box
            x1, y1 = int(w * 0.75), int(h * 0.75)
            
            logging.info(f"Fire detected with confidence {prediction:.4f}")
            return [["Fire", (x0, y0, x1, y1), float(prediction)]]
        
        return []
    
    except Exception as e:
        logging.error(f"Error in fire detection: {e}")
        return []

def draw_objects(frame):
    """Draw bounding boxes around detected objects."""
    global detections
    if detections:
        for class_name, bbox, score in detections:
            x0, y0, x1, y1 = bbox
            label = f"{class_name} %{int(score * 100)}"
            color = (0, 255, 0)  # Green by default
            
            # Use red color for fire detections
            if class_name.lower() == "fire":
                color = (0, 0, 255)  # Red in BGR
                
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cv2.putText(frame, label, (x0 + 5, y0 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame

class SimulatedCamera:
    """Simulated camera class for use when Pi camera is not available."""
    def __init__(self, width=1280, height=960, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.running = False
        self.encoder = None
        self.output = None
        
        # Try to use webcam if available, otherwise use generated frames
        self.use_webcam = False
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.use_webcam = True
                logging.info("Using webcam for video input")
            else:
                logging.info("Webcam not available, using generated frames")
        except:
            logging.info("Error accessing webcam, using generated frames")
    
    def start(self):
        self.running = True
        logging.info("Simulated camera started")
    
    def start_recording(self, encoder, output):
        self.encoder = encoder
        self.output = output
        # Start a thread that produces frames
        thread = threading.Thread(target=self._generate_frames, daemon=True)
        thread.start()
        logging.info("Simulated recording started")
    
    def _generate_frames(self):
        """Generate frames for the output."""
        global detections
        
        while self.running:
            if self.use_webcam:
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("Failed to get frame from webcam")
                    time.sleep(1/self.fps)
                    continue
                frame = cv2.resize(frame, (self.width, self.height))
                
                # Try to detect fire using our model if available
                if fire_model is not None:
                    fire_detections = detect_fire_in_frame(frame)
                    if fire_detections:
                        detections = fire_detections
                    elif int(time.time()) % 7 == 0:
                        # Clear detections occasionally
                        detections = []
                
            else:
                # Create a simple colored frame with timestamp
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                frame[:, :, :] = [64, 64, 64]  # Dark gray background
                
                # Add timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add simulated detection every few seconds
                if int(time.time()) % 5 == 0:
                    # If fire model available, add a fire detection occasionally
                    if fire_model is not None and random.random() < 0.3:
                        x0 = np.random.randint(100, self.width - 300)
                        y0 = np.random.randint(100, self.height - 300)
                        x1 = x0 + np.random.randint(100, 300)
                        y1 = y0 + np.random.randint(100, 300)
                        score = np.random.uniform(0.6, 0.95)
                        detections = [["Fire", (x0, y0, x1, y1), score]]
                    else:
                        # Simulate other object detection
                        class_id = np.random.randint(0, len(class_names))
                        x0 = np.random.randint(100, self.width - 300)
                        y0 = np.random.randint(100, self.height - 300)
                        x1 = x0 + np.random.randint(100, 300)
                        y1 = y0 + np.random.randint(100, 300)
                        score = np.random.uniform(0.6, 0.95)
                        detections = [[class_names[class_id], (x0, y0, x1, y1), score]]
                elif int(time.time()) % 7 == 0:
                    # Clear detections occasionally
                    detections = []
            
            # Draw objects on the frame
            frame = draw_objects(frame)
            
            # Convert to JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                self.output.write(jpeg.tobytes())
            
            # Simulate frame rate
            time.sleep(1/self.fps)
    
    def stop_recording(self):
        self.running = False
        if self.use_webcam and hasattr(self, 'cap'):
            self.cap.release()
        logging.info("Simulated recording stopped")

# Initialize either real or simulated camera
if PI_AVAILABLE:
    try:
        logging.info(f"Initializing Hailo with model: {hailo_model_path}")
        hailo = Hailo(hailo_model_path)
        model_h, model_w, _ = hailo.get_input_shape()
        logging.info(f"Model input shape: {model_w}x{model_h}")
        
        # Create Picamera2 instance and configure streams
        picam2 = Picamera2()
        main_size = (img_width, img_height)  # Main stream size
        lores_size = (320, 240)  # Lo-res stream size
        main = {'size': main_size, 'format': 'XRGB8888'}
        lores = {'size': lores_size, 'format': 'RGB888'}
        controls = {'FrameRate': frame_rate}

        video_config = picam2.create_preview_configuration(main, lores=lores, controls=controls)
        picam2.configure(video_config)
        
        def pi_draw_objects(request):
            """Draw bounding boxes for Pi camera."""
            global detections
            if detections:
                with MappedArray(request, "main") as m:
                    for class_name, bbox, score in detections:
                        x0, y0, x1, y1 = bbox
                        label = f"{class_name} %{int(score * 100)}"
                        cv2.rectangle(m.array, (x0, y0), (x1, y1), (0, 255, 0, 0), 2)
                        cv2.putText(m.array, label, (x0 + 5, y0 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0, 0), 1, cv2.LINE_AA)
        
        picam2.pre_callback = pi_draw_objects

        def process_frames():
            global detections
            while True:
                try:
                    frame = picam2.capture_array('lores')
                    # Resize frame to match model input if needed
                    if frame.shape[:2] != (model_h, model_w):
                        frame = cv2.resize(frame, (model_w, model_h))
                    results = hailo.run(frame)
                    detections = extract_detections(results, main_size[0], main_size[1], class_names, score_threshold)
                    
                    # Fire detection
                    fire_detections = detect_fire_in_frame(frame)
                    if fire_detections:
                        detections.extend(fire_detections)
                except Exception as e:
                    logging.error(f"Error in processing frame: {e}")
                    time.sleep(0.1)

        # Start background processing thread for object detection
        processing_thread = threading.Thread(target=process_frames, daemon=True)
        processing_thread.start()
        logging.info("Object detection processing thread started")

        # Start the camera and begin recording to the StreamingOutput
        picam2.start()
        logging.info("Camera started successfully")
        picam2.start_recording(JpegEncoder(), FileOutput(output))
        logging.info("Camera recording started")
        
        camera = picam2  # Assign to common variable
        
    except Exception as e:
        logging.error(f"Failed to initialize Pi camera: {e}")
        PI_AVAILABLE = False
        
if not PI_AVAILABLE:
    # Use simulated camera
    camera = SimulatedCamera(width=img_width, height=img_height, fps=frame_rate)
    camera.start()
    camera.start_recording(None, output)

def gen_frames():
    """Generator function that yields MJPEG frames."""
    while True:
        with output.condition:
            output.condition.wait()
            frame = output.frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index.html')
def index_html():
    return render_template('index.html')

@app.route('/stream.mjpg')
def video_feed():
    # Return a multipart response for MJPEG streaming
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        # Run Flask on all interfaces, port 8000, with threading enabled.
        app.run(host='0.0.0.0', port=8000, threaded=True)
    except KeyboardInterrupt:
        logging.info("Application shutting down")
    finally:
        if PI_AVAILABLE and 'hailo' in locals():
            hailo.close()
        elif not PI_AVAILABLE and 'camera' in locals():
            camera.stop_recording()