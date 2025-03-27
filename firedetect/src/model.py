import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

class WildfireDetectionModel:
    def __init__(self, img_size=(224, 224)):
        """
        Initialize the wildfire detection model.
        
        Args:
            img_size (tuple): Input image size (height, width)
        """
        self.img_size = img_size
        self.model = None
        
    def build_model(self, fine_tune=True):
        """
        Build the model architecture using transfer learning with MobileNetV2.
        
        Args:
            fine_tune (bool): Whether to fine-tune the base model
            
        Returns:
            tensorflow.keras.Model: Compiled model
        """
        # Load the pre-trained MobileNetV2 model
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze the base model if not fine-tuning
        base_model.trainable = fine_tune
        
        # Create the model architecture
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def load_model(self, model_path):
        """
        Load a saved model from disk.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            tensorflow.keras.Model: Loaded model
        """
        self.model = models.load_model(model_path)
        return self.model
    
    def save_model(self, model_path):
        """
        Save the current model to disk.
        
        Args:
            model_path (str): Path where to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Build or load a model first.")
        self.model.save(model_path)
    
    def predict(self, image):
        """
        Make a prediction on a preprocessed image.
        
        Args:
            image (numpy.ndarray): Preprocessed image array
            
        Returns:
            float: Probability of fire (0-1)
        """
        if self.model is None:
            raise ValueError("No model available. Build or load a model first.")
        return self.model.predict(image)[0][0]
    
    def get_model_summary(self):
        """
        Get a string representation of the model architecture.
        
        Returns:
            str: Model summary
        """
        if self.model is None:
            return "No model built yet."
        return self.model.summary() 