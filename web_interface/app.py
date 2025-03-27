import io
import logging
import threading
import time
import cv2
import numpy as np
from flask import Flask, render_template, Response
import os
import configparser

# Import TensorFlow and load_model
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
    logging.info("TensorFlow successfully imported")
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available")

# Import Picamera2 modules
from picamera2 import Picamera2, MappedArray
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
from picamera2.devices import Hailo

app = Flask(__name__)

# Global variable to store detections
detections = []

# Global fire model
fire_model = None

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
model_path = config['HailoModelPath']
labels_path = config['LabelsPath']
fire_model_path = config['FireModelPath']
score_threshold = float(config['ScoreThreshold'])
main_size = (int(config['ImageWidth']), int(config['ImageHeight']))
frame_rate = int(config['FrameRate'])

# Try to load the fire detection model
if TF_AVAILABLE:
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

def detect_fire_in_frame(frame):
    """Detect fire in a frame using the fire detection model."""
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

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()
    
    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

# Global streaming output instance
output = StreamingOutput()

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

def draw_objects(request):
    """Draw bounding boxes around detected objects."""
    global detections
    if detections:
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

# Load class names from the labels file
try:
    with open(labels_path, 'r', encoding="utf-8") as f:
        class_names = f.read().splitlines()
    logging.info(f"Loaded {len(class_names)} classes from {labels_path}")
except FileNotFoundError:
    class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "fire"]
    logging.warning(f"Labels file {labels_path} not found. Using default subset of classes.")

# Initialize Hailo device
try:
    logging.info(f"Initializing Hailo with model: {model_path}")
    hailo = Hailo(model_path)
    model_h, model_w, _ = hailo.get_input_shape()
    logging.info(f"Model input shape: {model_w}x{model_h}")
except Exception as e:
    logging.error(f"Failed to initialize Hailo: {e}")
    raise

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
    global detections
    while True:
        try:
            # Get frame from camera
            frame = picam2.capture_array('lores')
            
            # Run object detection with Hailo
            if frame.shape[:2] != (model_h, model_w):
                resized_frame = cv2.resize(frame, (model_w, model_h))
                results = hailo.run(resized_frame)
            else:
                results = hailo.run(frame)
                
            hailo_detections = extract_detections(results, main_size[0], main_size[1], class_names, score_threshold)
            
            # Run fire detection if model is available
            if fire_model is not None:
                fire_detections = detect_fire_in_frame(frame)
                
                # Combine detections
                all_detections = hailo_detections + fire_detections
                
                # Update global detections
                detections = all_detections
            else:
                detections = hailo_detections
                
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
    if 'hailo' in locals():
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

if __name__ == '__main__':
    try:
        # Run Flask on all interfaces, port 8000, with threading enabled.
        app.run(host='0.0.0.0', port=8000, threaded=True)
    finally:
        # Clean up
        if 'picam2' in locals():
            picam2.stop_recording()
            picam2.close()
        if 'hailo' in locals():
            hailo.close()