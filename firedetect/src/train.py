import os
import argparse
import tensorflow as tf
from datetime import datetime
from data import WildfireDataLoader
from model import WildfireDetectionModel
from visualize import WildfireVisualizer

def setup_logging(log_dir):
    """Set up TensorBoard logging."""
    log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )

def train_model(args):
    """
    Train the wildfire detection model.
    
    Args:
        args: Command line arguments containing training parameters
    """
    # Create all necessary directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.viz_dir, exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, 'train', 'fire'), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, 'train', 'no_fire'), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, 'validation', 'fire'), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, 'validation', 'no_fire'), exist_ok=True)
    
    # Initialize data loader
    data_loader = WildfireDataLoader(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    
    # Initialize visualizer
    visualizer = WildfireVisualizer(data_dir=args.data_dir)
    
    # Generate and save dataset visualizations
    print("Generating dataset visualizations...")
    visualizer.plot_dataset_distribution(
        save_path=os.path.join(args.viz_dir, 'dataset_distribution.png')
    )
    visualizer.plot_sample_images(
        num_samples=5,
        save_path=os.path.join(args.viz_dir, 'sample_images.png')
    )
    
    # Validate data directory
    if not data_loader.validate_data_directory():
        print("Error: Invalid data directory structure or no images found")
        print("Please add images to the following directories:")
        print(f"  {args.data_dir}/train/fire/")
        print(f"  {args.data_dir}/train/no_fire/")
        print(f"  {args.data_dir}/validation/fire/")
        print(f"  {args.data_dir}/validation/no_fire/")
        return
    
    # Get data generators
    train_generator, validation_generator = data_loader.load_data_generators()
    
    # Initialize and build model
    model = WildfireDetectionModel(img_size=(args.img_size, args.img_size))
    model.build_model(fine_tune=args.fine_tune)
    
    # Set up callbacks
    callbacks = [
        setup_logging(args.log_dir),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.model_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train the model
    print("\nStarting model training...")
    history = model.model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Update visualizer with trained model and history
    visualizer.model = model.model
    visualizer.history = history
    
    # Generate and save training visualizations
    print("\nGenerating training visualizations...")
    visualizer.plot_training_history(
        save_path=os.path.join(args.viz_dir, 'training_history.png')
    )
    
    # Generate predictions on validation set for confusion matrix
    val_predictions = model.model.predict(validation_generator)
    val_pred_classes = (val_predictions >= 0.5).astype(int)
    visualizer.plot_confusion_matrix(
        validation_generator.classes,
        val_pred_classes,
        save_path=os.path.join(args.viz_dir, 'confusion_matrix.png')
    )
    
    # Generate and save evaluation report
    visualizer.generate_evaluation_report(
        validation_generator.classes,
        val_pred_classes,
        save_path=os.path.join(args.viz_dir, 'evaluation_report.txt')
    )
    
    # Save the final model
    final_model_path = os.path.join(args.model_dir, 'final_model.h5')
    model.save_model(final_model_path)
    print(f"\nModel saved to {final_model_path}")
    
    # Print final metrics
    final_val_accuracy = history.history['val_accuracy'][-1]
    final_val_precision = history.history['val_precision'][-1]
    final_val_recall = history.history['val_recall'][-1]
    
    print("\nFinal Validation Metrics:")
    print(f"Accuracy: {final_val_accuracy:.4f}")
    print(f"Precision: {final_val_precision:.4f}")
    print(f"Recall: {final_val_recall:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train wildfire detection model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing fire and no_fire subdirectories')
    parser.add_argument('--model_dir', type=str, default='models',
                      help='Directory to save the trained model')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory for TensorBoard logs')
    parser.add_argument('--viz_dir', type=str, default='visualizations',
                      help='Directory for saving visualizations')
    parser.add_argument('--img_size', type=int, default=224,
                      help='Image size (width and height)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--fine_tune', action='store_true',
                      help='Whether to fine-tune the base model')
    
    args = parser.parse_args()
    
    # Train the model
    train_model(args)

if __name__ == '__main__':
    main() 