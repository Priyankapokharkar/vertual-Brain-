import os
import sys
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from config import Config
from models.disease_detector import DiseaseDetector

def prepare_dataset_directories():
    """Organize the dataset into standard Healthy/Tumor directories for Keras"""
    import shutil
    
    config = Config()
    source_dir = config.DATA_DIR / 'datasets' / 'archive (4)'
    target_dir = config.DATA_DIR / 'training_data'
    
    if target_dir.exists():
        print(f"Dataset already prepared at {target_dir}")
        return target_dir
        
    print(f"Preparing dataset at {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Create class directories
    healthy_dir = target_dir / 'Healthy'
    tumor_dir = target_dir / 'Tumor'
    healthy_dir.mkdir(exist_ok=True)
    tumor_dir.mkdir(exist_ok=True)
    
    # Copy Normal images to Healthy
    normal_source = source_dir / 'Normal'
    if normal_source.exists():
        print("Copying Normal -> Healthy")
        for file in normal_source.glob('*.jpg'):
            shutil.copy(file, healthy_dir / file.name)
            
    # Copy all BT_ images to Tumor
    for class_name in ['BT_glioma', 'BT_meningioma', 'BT_pituitary']:
        class_source = source_dir / class_name
        if class_source.exists():
            print(f"Copying {class_name} -> Tumor")
            for file in class_source.glob('*.jpg'):
                # Add prefix to avoid filename collisions
                new_name = f"{class_name}_{file.name}"
                shutil.copy(file, tumor_dir / new_name)
                
    print("Dataset preparation complete.")
    return target_dir

def train_model():
    """Train the disease detector model"""
    # 1. Prepare data
    data_dir = prepare_dataset_directories()
    
    print("\n--- Starting Model Training ---")
    
    # 2. Setup standard Keras data generators
    batch_size = 32
    img_height = 128
    img_width = 128
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        color_mode="grayscale", # Match our 1 channel model
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical' # Ensure it one-hot encodes for our 2 output nodes
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        color_mode="grayscale",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    class_names = train_ds.class_names
    print(f"Discovered classes: {class_names}")
    
    # Configure dataset for performance and normalize
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (x / 255.0, y)).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (x / 255.0, y)).cache().prefetch(buffer_size=AUTOTUNE)
    
    # 3. Initialize model
    detector = DiseaseDetector()
    model = detector.build_model(input_shape=(img_height, img_width, 1))
    
    # 4. Callbacks
    config = Config()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(config.DISEASE_MODEL_PATH),
            save_best_only=True,
            monitor='val_accuracy'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )
    ]
    
    # 5. Train
    print("\nTraining for 10 epochs (early stopping enabled)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks
    )
    
    print(f"\nTraining complete. Best model saved to {config.DISEASE_MODEL_PATH}")

if __name__ == '__main__':
    train_model()
