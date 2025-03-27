import io
import logging
import threading
import time
import cv2
import numpy as np
from flask import Flask, render_template, Response
import os
import configparser
import sys

# Configure logging - increase level to make fire detection more visible
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add firedetect/src to Python path to import from there directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 
                               '../firedetect/src')))

# Import TensorFlow directly from system rather than through another module
fire_model = None
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    tf.get_logger().setLevel('ERROR')  # Suppress TF INFO messages
    TF_AVAILABLE = True
    logging.info("TensorFlow successfully imported")
except ImportError:
    logging.warning("TensorFlow not available, will use mock detection")

# Import Picamera2 modules
from picamera2 import Picamera2, MappedArray
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
try:
    from picamera2.devices import Hailo
    HAILO_AVAILABLE = True
    logging.info("Hailo module available")
except ImportError:
    HAILO_AVAILABLE = False
    logging.warning("Hailo module not available, will use mock detection")

app = Flask(__name__)

# Global variable to store detections
detections = []

# Global streaming output instance
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()
    
    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

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
        'FireDetectionThreshold': '0.5',  # Add separate threshold for fire
        'SimulateFireDetection': 'false',  # Add option to simulate fire
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
fire_threshold = float(config.get('FireDetectionThreshold', '0.5'))
simulate_fire = config.get('SimulateFireDetection', 'false').lower() == 'true'
main_size = (int(config['ImageWidth']), int(config['ImageHeight']))
frame_rate = int(config['FrameRate'])

# Global streaming output instance
output = StreamingOutput()

# Try to load the fire detection model
if TF_AVAILABLE:
    try:
        fire_model_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                      fire_model_path))
        if os.path.exists(fire_model_path):
            logging.info(f"Loading fire model from {fire_model_path}")
            # Use compile=False to avoid compatibility issues
            fire_model = load_model(fire_model_path, compile=False)
            logging.info(f"Successfully loaded fire detection model")
        else:
            logging.warning(f"Fire model not found at {fire_model_path}")
    except Exception as e:
        logging.error(f"Error loading fire detection model: {e}")
        fire_model = None

def preprocess_image(image):
    """Preprocess an image for the fire detection model."""
    try:
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Handle grayscale or other formats
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize to model input size (224x224 for MobileNetV2)
        image_resized = cv2.resize(image_rgb, (224, 224))
        
        # Normalize pixel values to 0-1
        image_normalized = image_resized.astype('float32') / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        return None

def detect_fire_in_frame(frame):
    """Detect fire in a frame using the fire detection model."""
    global fire_model, simulate_fire
    
    # If simulation is enabled, periodically return a fire detection
    if simulate_fire and int(time.time()) % 10 == 0:
        h, w = frame.shape[:2]
        x0, y0 = int(w * 0.25), int(h * 0.25)
        x1, y1 = int(w * 0.75), int(h * 0.75)
        logging.warning("SIMULATED FIRE DETECTED")
        return [["Fire", (x0, y0, x1, y1), 0.85]]
    
    if fire_model is None:
        return []
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(frame)
        if processed_image is None:
            return []
        
        # Get prediction
        prediction = fire_model.predict(processed_image, verbose=0)[0][0]
        
        # Check if fire detected based on threshold
        if prediction >= fire_threshold:
            # Create a detection with the same format as other detections
            h, w = frame.shape[:2]
            x0, y0 = int(w * 0.25), int(h * 0.25)
            x1, y1 = int(w * 0.75), int(h * 0.75)
            
            logging.warning(f"FIRE DETECTED with confidence {prediction:.4f}")
            return [["Fire", (x0, y0, x1, y1), float(prediction)]]
        
        return []
    except Exception as e:
        logging.error(f"Error in fire detection: {e}")
        return []

def extract_detections(hailo_output, w, h, class_names, threshold=0.5):
    """Extract detections from the HailoRT-postprocess output."""
    results = []
    try:
        for class_id, dets in enumerate(hailo_output):
            if class_id >= len(class_names):
                continue  # Skip if class_id is out of range
                
            for detection in dets:
                if len(detection) < 5:  # Check if detection has enough elements
                    continue
                    
                score = detection[4]
                if score >= threshold:
                    y0, x0, y1, x1 = detection[:4]
                    bbox = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
                    results.append([class_names[class_id], bbox, score])
    except Exception as e:
        logging.error(f"Error extracting detections: {e}")
    return results

def draw_objects(request):
    """Draw bounding boxes around detected objects."""
    global detections
    if detections:
        try:
            with MappedArray(request, "main") as m:
                for class_name, bbox, score in detections:
                    x0, y0, x1, y1 = bbox
                    label = f"{class_name} %{int(score * 100)}"
                    
                    # Use red color for fire detections, green for others
                    color = (0, 255, 0, 0)  # Green in XRGB format
                    if class_name.lower() == "fire":
                        color = (0, 0, 255, 0)  # Red in XRGB format
                    
                    cv2.rectangle(m.array, (x0, y0), (x1, y1), color, 2)
                    cv2.putText(m.array, label, (x0 + 5, y0 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        except Exception as e:
            logging.error(f"Error drawing objects: {e}")

# Load class names from the labels file
try:
    labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), labels_path)
    with open(labels_path, 'r', encoding="utf-8") as f:
        class_names = f.read().splitlines()
    logging.info(f"Loaded {len(class_names)} classes from {labels_path}")
except FileNotFoundError:
    class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "fire"]
    logging.warning(f"Labels file {labels_path} not found. Using default subset of classes.")

# Initialize Hailo device if available
hailo = None
model_h, model_w = 640, 640  # Default values
if HAILO_AVAILABLE:
    try:
        logging.info(f"Initializing Hailo with model: {hailo_model_path}")
        hailo = Hailo(hailo_model_path)
        model_h, model_w, _ = hailo.get_input_shape()
        logging.info(f"Model input shape: {model_w}x{model_h}")
    except Exception as e:
        logging.error(f"Failed to initialize Hailo: {e}")
        hailo = None

# Create Picamera2 instance and configure streams
picam2 = Picamera2()
lores_size = (320, 240)  # Lo-res stream size for detection
main = {'size': main_size, 'format': 'XRGB8888'}
lores = {'size': lores_size, 'format': 'RGB888'}
controls = {'FrameRate': frame_rate}

video_config = picam2.create_preview_configuration(main, lores=lores, controls=controls)
picam2.configure(video_config)
picam2.pre_callback = draw_objects

def process_frames():
    global detections, hailo, fire_model
    
    # Log what detection methods are available
    if fire_model is not None:
        logging.info("Fire detection model loaded and will be used")
    else:
        logging.warning("Fire detection model not available")
        
    if hailo is not None:
        logging.info("Hailo detection available and will be used")
    else:
        logging.warning("Hailo detection not available")
    
    # Force testing on startup to verify fire detection pipeline
    startup_test_done = False
    
    while True:
        try:
            # Get frame from camera
            frame = picam2.capture_array('lores')
            
            # Run one-time test on startup to verify detection pipeline
            if not startup_test_done and fire_model is not None:
                try:
                    # Create a small manually crafted fire image for testing
                    test_img = np.zeros((320, 240, 3), dtype=np.uint8)
                    test_img[80:160, 80:160, 0] = 0     # Blue channel
                    test_img[80:160, 80:160, 1] = 0     # Green channel
                    test_img[80:160, 80:160, 2] = 255   # Red channel (fire color)
                    
                    # Try to predict on this test image
                    processed_img = preprocess_image(test_img)
                    if processed_img is not None:
                        pred = fire_model.predict(processed_img, verbose=0)[0][0]
                        logging.warning(f"Fire detection test result: {pred:.4f}")
                    
                    startup_test_done = True
                except Exception as e:
                    logging.error(f"Fire detection test failed: {e}")
                    startup_test_done = True
            
            # Always try fire detection first
            fire_detections = []
            try:
                fire_detections = detect_fire_in_frame(frame)
                if fire_detections:
                    # Log clearly with easily visible markers
                    logging.warning(f"!!! FIRE DETECTED !!! Confidence: {fire_detections[0][2]:.4f}")
                    # Update the global detections immediately with fire info
                    detections = fire_detections
                    time.sleep(0.1)
                    continue
            except Exception as e:
                logging.error(f"Error in fire detection: {e}")
            
            # If no fire detected and Hailo is available, use it
            hailo_detections = []
            if hailo is not None:
                try:
                    # Make sure we have a clean RGB image of the correct size
                    if frame.shape[:2] != (model_h, model_w):
                        resized_frame = cv2.resize(frame, (model_w, model_h))
                    else:
                        resized_frame = frame.copy()
                    
                    # Ensure correct format for Hailo
                    if len(resized_frame.shape) != 3 or resized_frame.shape[2] != 3:
                        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2RGB)
                    
                    # Ensure contiguity
                    resized_frame = np.ascontiguousarray(resized_frame)
                    
                    # Run detection with Hailo
                    results = hailo.run(resized_frame)
                    hailo_detections = extract_detections(results, main_size[0], main_size[1], 
                                                         class_names, score_threshold)
                except Exception as e:
                    logging.error(f"Error in Hailo detection: {e}")
            
            # If Hailo provided detections, use them
            if hailo_detections:
                detections = hailo_detections
            elif simulate_fire and int(time.time()) % 10 == 0:
                # Simulate fire detection periodically if enabled
                h, w = frame.shape[:2]
                x0, y0 = int(w * 0.25), int(h * 0.25)
                x1, y1 = int(w * 0.75), int(h * 0.75)
                detections = [["Fire", (x0, y0, x1, y1), 0.85]]
                logging.warning("SIMULATED FIRE DETECTED")
            else:
                # No detections found and simulation not active, clear detections
                detections = []
                
        except Exception as e:
            logging.error(f"Error in processing frame: {e}")
            time.sleep(0.1)

# Start background processing thread for object detection
processing_thread = threading.Thread(target=process_frames, daemon=True)
processing_thread.start()
logging.info("Object detection processing thread started")

# Start the camera and begin recording to the StreamingOutput
try:
    picam2.start()
    logging.info("Camera started successfully")
    picam2.start_recording(JpegEncoder(), FileOutput(output))
    logging.info("Camera recording started")
except Exception as e:
    logging.error(f"Failed to start camera: {e}")
    if hailo is not None:
        hailo.close()
    raise

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

# Add endpoint to force fire detection for testing
@app.route('/simulate_fire/<state>')
def simulate_fire_detection(state):
    global simulate_fire
    simulate_fire = (state.lower() == 'on')
    message = f"Fire simulation {'ON' if simulate_fire else 'OFF'}"
    logging.warning(message)
    return message

if __name__ == '__main__':
    try:
        # Run Flask on all interfaces, port 8000, with threading enabled.
        app.run(host='0.0.0.0', port=8000, threaded=True)
    except KeyboardInterrupt:
        logging.info("Application shutting down")
    finally:
        # Clean up
        if 'picam2' in locals():
            try:
                picam2.stop_recording()
                picam2.close()
            except:
                pass
        if hailo is not None:
            try:
                hailo.close()
            except:
                pass