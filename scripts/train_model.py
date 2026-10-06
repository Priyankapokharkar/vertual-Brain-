import os
import glob
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, RandomFlip, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import numpy as np
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

# Configuration
DATA_DIR = r"d:\Projects 2025\Final Year projects\Vertual brain\data\training_unified"
MODEL_SAVE_PATH = r"d:\Projects 2025\Final Year projects\Vertual brain\models\pretrained\multi_disease_detector.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30

def get_data_lists():
    class_names = sorted(os.listdir(DATA_DIR))
    file_paths = []
    labels = []
    
    for i, cls in enumerate(class_names):
        cls_path = os.path.join(DATA_DIR, cls)
        files = glob.glob(os.path.join(cls_path, "*.*"))
        file_paths.extend(files)
        labels.extend([i] * len(files))
    
    return file_paths, labels, class_names

def oversample_list(file_paths, labels):
    unique_labels = np.unique(labels)
    target_count = 5000
    
    new_paths = []
    new_labels = []
    
    for label in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == label]
        cls_files = [file_paths[i] for i in idx]
        
        current_count = len(cls_files)
        if current_count < target_count:
            multiplier = int(np.ceil(target_count / current_count))
            oversampled = (cls_files * multiplier)[:target_count]
        else:
            oversampled = cls_files[:target_count]
            
        new_paths.extend(oversampled)
        new_labels.extend([label] * len(oversampled))
        
    return new_paths, new_labels

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.resnet_v2.preprocess_input(img)
    return img, tf.one_hot(label, 9)

def train():
    # 1. Prepare Data securely (Split then Oversample)
    print("Collecting data and performing secure train-test split...")
    file_paths, labels, class_names = get_data_lists()
    num_classes = len(class_names)
    
    # Split first!
    X_train_raw, X_val, y_train_raw, y_val = train_test_split(file_paths, labels, test_size=0.15, stratify=labels, random_state=42)
    
    # Oversample ONLY the training set
    print("Oversampling training set...")
    X_train, y_train = oversample_list(X_train_raw, y_train_raw)
    
    print(f"Final training size: {len(X_train)}")
    print(f"Final validation size: {len(X_val)}")

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(len(X_train)).map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # 2. Augmentation Layer
    augmentation = tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.2),
        RandomZoom(0.2),
        RandomContrast(0.2)
    ])

    # 3. Model Architecture
    print("Building ResNet50V2 model with advanced head...")
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.4)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)

    # 4. Phase 1: Warmup
    print("Starting Training Phase 1: Warmup classifier...")
    # Add LabelSmoothing for better generalization
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-4), loss=loss_fn, metrics=['accuracy'])
    
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.2, patience=3, monitor='val_accuracy', min_lr=1e-7)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

    # 5. Phase 2: Deep Fine-tuning
    print("Starting Training Phase 2: Deep Fine-tuning (last 100 layers)...")
    base_model.trainable = True
    for layer in base_model.layers[:-100]:
        layer.trainable = False
        
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=loss_fn, metrics=['accuracy'])
    
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    print(f"Model trained and saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
