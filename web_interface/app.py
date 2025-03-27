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

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add firedetect/src to Python path to import from there directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../firedetect/src')))

# Try to import YOLO for fire detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logging.info("YOLOv8 successfully imported")
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLOv8 not available, will fall back to simulation mode")

# Fallback to trying TensorFlow for the older model
if not YOLO_AVAILABLE:
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        tf.get_logger().setLevel('ERROR')  # Suppress TF INFO messages
        TF_AVAILABLE = True
        logging.info("TensorFlow successfully imported (fallback mode)")
    except ImportError:
        TF_AVAILABLE = False
        logging.warning("Neither YOLOv8 nor TensorFlow available, will use simulation mode")

# Import Picamera2 modules
try:
    from picamera2 import Picamera2, MappedArray
    from picamera2.encoders import JpegEncoder
    from picamera2.outputs import FileOutput
    PICAMERA_AVAILABLE = True
    logging.info("Picamera2 successfully imported")
except ImportError:
    PICAMERA_AVAILABLE = False
    logging.warning("Picamera2 modules not available - running in simulation mode")

app = Flask(__name__)

# Global variable to store detections
detections = []

# Global model variables
yolo_model = None
tf_model = None

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
        'YoloModelPath': '../firedetect/models/yolov8_fire_detection.pt',
        'OldModelPath': '../firedetect/models/best_model.h5',
        'ScoreThreshold': '0.5',
        'FireDetectionThreshold': '0.5',
        'SimulateFireDetection': 'false',
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
yolo_model_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                              config['YoloModelPath']))
old_model_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                             config['OldModelPath']))
score_threshold = float(config['ScoreThreshold'])
fire_threshold = float(config.get('FireDetectionThreshold', '0.5'))
simulate_fire = config.get('SimulateFireDetection', 'false').lower() == 'true'
main_size = (int(config['ImageWidth']), int(config['ImageHeight']))
frame_rate = int(config['FrameRate'])

# Global streaming output instance
output = StreamingOutput()

# Load the YOLO model if available
if YOLO_AVAILABLE:
    try:
        if os.path.exists(yolo_model_path):
            logging.info(f"Loading YOLOv8 model from {yolo_model_path}")
            yolo_model = YOLO(yolo_model_path)
            logging.info("Successfully loaded YOLOv8 model for fire detection")
        else:
            # If the YOLO model doesn't exist, use the default YOLOv8 nano model
            logging.warning(f"YOLOv8 model not found at {yolo_model_path}, using default model")
            yolo_model = YOLO("yolov8n.pt")
            logging.info("Loaded default YOLOv8n model")
    except Exception as e:
        logging.error(f"Error loading YOLOv8 model: {e}")
        yolo_model = None

# If YOLO model failed to load, try to load the old TensorFlow model
if yolo_model is None and TF_AVAILABLE:
    try:
        if os.path.exists(old_model_path):
            logging.info(f"Loading TensorFlow model from {old_model_path}")
            tf_model = load_model(old_model_path, compile=False)
            logging.info("Successfully loaded TensorFlow model for fire detection")
        else:
            logging.warning(f"TensorFlow model not found at {old_model_path}")
    except Exception as e:
        logging.error(f"Error loading TensorFlow model: {e}")
        tf_model = None

def preprocess_image_tf(image):
    """Preprocess an image for the TensorFlow model."""
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
        logging.error(f"Error preprocessing image for TensorFlow: {e}")
        return None

def detect_fire_in_frame(frame):
    """
    Detect fire in a frame using the available model.
    Returns list of detections in format: [[class_name, bbox, confidence], ...]
    """
    global yolo_model, tf_model, simulate_fire
    
    # If simulation is enabled, periodically return a fire detection
    if simulate_fire and int(time.time()) % 10 == 0:
        h, w = frame.shape[:2]
        x0, y0 = int(w * 0.25), int(h * 0.25)
        x1, y1 = int(w * 0.75), int(h * 0.75)
        logging.warning("SIMULATED FIRE DETECTED")
        return [["Fire", (x0, y0, x1, y1), 0.85]]
    
    # First try using YOLOv8 model
    if yolo_model is not None:
        try:
            # Run YOLO detection
            results = yolo_model.predict(source=frame, conf=fire_threshold, verbose=False)
            
            if results and len(results) > 0:
                # Process results
                result = results[0]  # Get first (and only) result
                boxes = result.boxes
                
                fire_detections = []
                for i, box in enumerate(boxes):
                    # Get class, confidence and box coordinates
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Class 0 should be 'fire' based on our training
                    if cls_id == 0:  # Fire class
                        fire_detections.append(["Fire", (x1, y1, x2, y2), conf])
                        logging.warning(f"YOLO FIRE DETECTED with confidence {conf:.4f}")
                
                return fire_detections
            
            return []
        
        except Exception as e:
            logging.error(f"Error in YOLOv8 fire detection: {e}")
            # Fall back to TensorFlow model if available
    
    # If YOLO fails or is not available, try the TensorFlow model
    if tf_model is not None:
        try:
            # Preprocess the image
            processed_image = preprocess_image_tf(frame)
            if processed_image is None:
                return []
            
            # Get prediction
            prediction = tf_model.predict(processed_image, verbose=0)[0][0]
            
            # Check if fire detected based on threshold
            if prediction >= fire_threshold:
                # Create a detection with the same format as other detections
                h, w = frame.shape[:2]
                x0, y0 = int(w * 0.25), int(h * 0.25)
                x1, y1 = int(w * 0.75), int(h * 0.75)
                
                logging.warning(f"TF FIRE DETECTED with confidence {prediction:.4f}")
                return [["Fire", (x0, y0, x1, y1), float(prediction)]]
        
        except Exception as e:
            logging.error(f"Error in TensorFlow fire detection: {e}")
    
    # No fire detected or models failed
    return []

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

# Initialize camera based on what's available
if PICAMERA_AVAILABLE:
    try:
        # Create Picamera2 instance and configure streams
        picam2 = Picamera2()
        lores_size = (320, 240)  # Lo-res stream size for detection
        main = {'size': main_size, 'format': 'XRGB8888'}
        lores = {'size': lores_size, 'format': 'RGB888'}
        controls = {'FrameRate': frame_rate}

        video_config = picam2.create_preview_configuration(main, lores=lores, controls=controls)
        picam2.configure(video_config)
        picam2.pre_callback = draw_objects
        
        logging.info("Picamera2 configured successfully")
    except Exception as e:
        logging.error(f"Failed to configure Picamera2: {e}")
        PICAMERA_AVAILABLE = False

class SimulatedCamera:
    """Simulated camera class for use when Pi camera is not available."""
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.running = False
        self.cap = None
        self.output = None
        
        # Try to use webcam if available
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                logging.info("Using webcam for video input")
            else:
                logging.warning("Webcam not available, using generated frames")
                self.cap = None
        except Exception as e:
            logging.warning(f"Error accessing webcam ({e}), using generated frames")
            self.cap = None
    
    def start(self):
        self.running = True
        logging.info("Simulated camera started")
    
    def start_recording(self, encoder, output):
        self.output = output
        # Start a thread that produces frames
        thread = threading.Thread(target=self._generate_frames, daemon=True)
        thread.start()
        logging.info("Simulated recording started")
    
    def _generate_frames(self):
        """Generate frames for the output."""
        global detections
        
        while self.running:
            try:
                if self.cap is not None and self.cap.isOpened():
                    # Get frame from webcam
                    ret, frame = self.cap.read()
                    if not ret:
                        # If webcam fails, fall back to generated frame
                        logging.warning("Failed to get frame from webcam")
                        frame = self._create_simulated_frame()
                    else:
                        # Resize frame to match expected dimensions
                        frame = cv2.resize(frame, (self.width, self.height))
                else:
                    # Create a simulated frame
                    frame = self._create_simulated_frame()
                
                # Try to detect fire in the frame
                fire_detections = detect_fire_in_frame(frame)
                if fire_detections:
                    detections = fire_detections
                elif int(time.time()) % 7 == 0:
                    # Clear detections occasionally
                    detections = []
                
                # Draw objects on the frame
                for detection in detections:
                    class_name, bbox, score = detection
                    x0, y0, x1, y1 = bbox
                    label = f"{class_name} %{int(score * 100)}"
                    
                    # Use red color for fire detections, green for others
                    color = (0, 255, 0)  # Green in BGR
                    if class_name.lower() == "fire":
                        color = (0, 0, 255)  # Red in BGR
                    
                    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                    cv2.putText(frame, label, (x0 + 5, y0 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                
                # Convert to JPEG
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    self.output.write(jpeg.tobytes())
                
                # Control frame rate
                time.sleep(1/self.fps)
                
            except Exception as e:
                logging.error(f"Error in simulated camera: {e}")
                time.sleep(0.5)
    
    def _create_simulated_frame(self):
        """Create a simulated frame with timestamp."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :, :] = [64, 64, 64]  # Dark gray background
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add message about simulation mode
        cv2.putText(frame, "Simulation Mode", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if yolo_model is not None:
            cv2.putText(frame, "YOLOv8 Loaded", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif tf_model is not None:
            cv2.putText(frame, "TensorFlow Model Loaded", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Detection Model", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame
    
    def stop_recording(self):
        self.running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        logging.info("Simulated recording stopped")

def process_frames():
    """Process frames from the Picamera2 for fire detection."""
    global detections, yolo_model, tf_model
    
    # Log what detection methods are available
    if yolo_model is not None:
        logging.info("YOLOv8 model loaded and will be used for fire detection")
    elif tf_model is not None:
        logging.info("TensorFlow model loaded and will be used for fire detection")
    else:
        logging.warning("No detection models available, will use simulation only")
    
    # Force testing on startup to verify fire detection pipeline
    startup_test_done = False
    
    while True:
        try:
            # Get frame from camera
            frame = picam2.capture_array('lores')
            
            # Run one-time test on startup to verify detection pipeline
            if not startup_test_done:
                try:
                    # Create a test image with a red rectangle (fire-like)
                    test_img = np.zeros((320, 240, 3), dtype=np.uint8)
                    test_img[80:160, 80:160, 0] = 0     # Blue channel
                    test_img[80:160, 80:160, 1] = 0     # Green channel
                    test_img[80:160, 80:160, 2] = 255   # Red channel (fire color)
                    
                    # Run detection on test image
                    test_detections = detect_fire_in_frame(test_img)
                    if test_detections:
                        logging.warning(f"Fire detection test PASSED: {test_detections[0][2]:.4f}")
                    else:
                        logging.warning("Fire detection test FAILED: no fire detected in test image")
                    
                    # Enable simulation briefly to show it works
                    global simulate_fire
                    simulate_fire = True
                    logging.warning("Enabling fire simulation for 15 seconds to verify display")
                    
                    # Create a thread to disable simulation after 15 seconds
                    def disable_simulation():
                        time.sleep(15)
                        global simulate_fire
                        simulate_fire = False
                        logging.warning("Fire simulation disabled after timeout")
                    
                    sim_thread = threading.Thread(target=disable_simulation, daemon=True)
                    sim_thread.start()
                    
                    startup_test_done = True
                except Exception as e:
                    logging.error(f"Fire detection test failed: {e}")
                    startup_test_done = True
            
            # Try fire detection
            fire_detections = detect_fire_in_frame(frame)
            if fire_detections:
                # Update the global detections immediately with fire info
                detections = fire_detections
            else:
                # No detections found
                detections = []
                
        except Exception as e:
            logging.error(f"Error in processing frame: {e}")
            time.sleep(0.1)

# Start the system
if PICAMERA_AVAILABLE:
    try:
        # Start background processing thread for object detection
        processing_thread = threading.Thread(target=process_frames, daemon=True)
        processing_thread.start()
        logging.info("Object detection processing thread started")

        # Start the camera and begin recording to the StreamingOutput
        picam2.start()
        logging.info("Camera started successfully")
        picam2.start_recording(JpegEncoder(), FileOutput(output))
        logging.info("Camera recording started")
        
        # Set camera variable for later use
        camera = picam2
    except Exception as e:
        logging.error(f"Failed to start Picamera2: {e}")
        PICAMERA_AVAILABLE = False

if not PICAMERA_AVAILABLE:
    # Use simulated camera
    camera = SimulatedCamera(width=main_size[0], height=main_size[1], fps=frame_rate)
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

@app.route('/simulate_fire/<state>')
def simulate_fire_detection(state):
    global simulate_fire
    simulate_fire = (state.lower() == 'on')
    message = f"Fire simulation {'ON' if simulate_fire else 'OFF'}"
    logging.warning(message)
    return message

@app.route('/info')
def system_info():
    """Return information about the system status."""
    info = {
        "yolov8_available": YOLO_AVAILABLE,
        "tensorflow_available": TF_AVAILABLE,
        "picamera_available": PICAMERA_AVAILABLE,
        "yolo_model_loaded": yolo_model is not None,
        "tf_model_loaded": tf_model is not None,
        "simulation_mode": simulate_fire
    }
    
    result = "<h1>System Status</h1><ul>"
    for key, value in info.items():
        result += f"<li>{key}: {'Yes' if value else 'No'}</li>"
    result += "</ul>"
    
    return result

if __name__ == '__main__':
    try:
        # Run Flask on all interfaces, port 8000, with threading enabled.
        app.run(host='0.0.0.0', port=8000, threaded=True)
    except KeyboardInterrupt:
        logging.info("Application shutting down")
    finally:
        # Clean up
        if PICAMERA_AVAILABLE and 'camera' in locals():
            try:
                camera.stop_recording()
                if hasattr(camera, 'close'):
                    camera.close()
            except:
                pass
        elif not PICAMERA_AVAILABLE and 'camera' in locals():
            camera.stop_recording()