import io
import logging
import threading
import time
import cv2
import numpy as np
from flask import Flask, render_template, Response

from picamera2 import Picamera2, MappedArray
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
from picamera2.devices import Hailo

app = Flask(__name__)

# Global variable to store detections
detections = []

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
                cv2.rectangle(m.array, (x0, y0), (x1, y1), (0, 255, 0, 0), 2)
                cv2.putText(m.array, label, (x0 + 5, y0 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0, 0), 1, cv2.LINE_AA)

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

# Path to model and labels
model_path = "/usr/share/hailo-models/yolov8s_h8l.hef"
labels_path = "coco.txt"
score_threshold = 0.5

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load class names from the labels file
try:
    with open(labels_path, 'r', encoding="utf-8") as f:
        class_names = f.read().splitlines()
    logging.info(f"Loaded {len(class_names)} classes from {labels_path}")
except FileNotFoundError:
    class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
    logging.warning(f"Labels file {labels_path} not found. Using default subset of classes.")

# Initialize Hailo device and camera
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
main_size = (1280, 960)  # Main stream size
lores_size = (320, 240)  # Lo-res stream size
main = {'size': main_size, 'format': 'XRGB8888'}
lores = {'size': lores_size, 'format': 'RGB888'}
controls = {'FrameRate': 30}

video_config = picam2.create_preview_configuration(main, lores=lores, controls=controls)
picam2.configure(video_config)
picam2.pre_callback = draw_objects

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

@app.route('/index.html')
def index_html():
    return render_template('index.html')


@app.route('/stream.mjpg')
def video_feed():
    # Return a multipart response for MJPEG streaming
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run Flask on all interfaces, port 8000, with threading enabled.
    app.run(host='0.0.0.0', port=8000, threaded=True)