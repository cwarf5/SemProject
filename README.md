# Wildfire Detection AI/ML System

An AI-powered system for detecting wildfires in images and video streams using deep learning.

## Project Overview

This system uses computer vision and deep learning to detect the presence of wildfires in images and video streams. The project is implemented in three phases:

1. **Phase 1:** Core AI/ML model for image-based fire detection
2. **Phase 2:** Real-time webcam integration
3. **Phase 3:** Raspberry Pi deployment
4. **Phase 4:** Web Interface Integration

## Features

- Transfer learning using MobileNetV2 for efficient fire detection
- Real-time webcam monitoring with visual alerts
- Comprehensive data visualization and model performance analysis
- Interactive dataset verification and cleaning tools
- Automated model evaluation and reporting

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd firedetect
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
firedetect/
├── data/
│   ├── fire/         # Fire images for training
│   ├── no_fire/      # Non-fire images for training
│   └── review/       # Images requiring manual review
├── src/
│   ├── data.py           # Data loading and preprocessing
│   ├── model.py          # Neural network model definition
│   ├── train.py          # Training script
│   ├── inference.py      # Inference module
│   └── visualize.py      # Visualization tools
├── models/           # Saved model files
├── logs/             # Training and inference logs
├── visualizations/   # Generated plots and visualizations
├── requirements.txt
└── README.md
web_interface/
├── app.py            # Web application for camera streaming and detection
├── static/           # Static files for web interface
└── templates/        # HTML templates for web interface
```

## Usage

### Data Preparation

For proper training, organize your dataset:
- Place fire images in `data/fire/`
- Place non-fire images in `data/no_fire/`

Ensure your dataset is properly categorized and labeled before proceeding with training.

### Training the Model

```bash
python src/train.py --data_dir data/ --epochs 50 --batch_size 32
```

The training process includes:
- Automatic data augmentation
- Transfer learning with MobileNetV2
- Early stopping to prevent overfitting
- Model checkpointing to save the best model
- Comprehensive visualization generation

### Visualizations

During training, the system automatically generates:
- Dataset distribution plots
- Sample image grids
- Training/validation metrics plots
- Confusion matrices
- Detailed evaluation reports

All visualizations are saved in the `visualizations/` directory.

### Running Inference on Images

```bash
python src/inference.py --model_path models/best_model.h5 --image_path path/to/image.jpg
```

### Real-time Webcam Detection

To use the web interface for real-time wildfire detection:

```bash
cd web_interface
python app.py
```

This will start a web server that provides:
- Real-time camera streaming
- Fire detection on camera feed
- Visual alerts when fire is detected

## Model Architecture

The system uses a transfer learning approach based on MobileNetV2:
- Pre-trained on ImageNet
- Fine-tuned for wildfire detection
- Additional custom layers for binary classification
- Optimized for real-time performance

## Performance Metrics

The model is evaluated on:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Data Requirements

- Place fire images in `data/fire/`
- Place non-fire (normal forest) images in `data/no_fire/`
- Supported formats: JPG, PNG
- Recommended minimum dataset size: 1000 images per class

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Recent Changes

- Added comprehensive data visualization tools
- Implemented interactive dataset verification system
- Enhanced model evaluation with detailed metrics
- Added automatic visualization generation during training
- Improved documentation and code organization