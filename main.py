import os
import logging
import glob
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
from tqdm import tqdm
import SimpleITK as sitk

from config import Config  # Import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
np.random.seed(Config.SEED)
tf.random.set_seed(Config.SEED)

def load_mhd_file(file_path):
    """Load .mhd file and return CT scan array, origin, and spacing."""
    try:
        itkimage = sitk.ReadImage(file_path)
        ct_scan = sitk.GetArrayFromImage(itkimage)
        origin = np.array(itkimage.GetOrigin())
        spacing = np.array(itkimage.GetSpacing())
        return ct_scan, origin, spacing
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None, None, None

def convert_to_hounsfield(ct_scan):
    """Convert CT scan to Hounsfield units."""
    if ct_scan is not None:
        ct_scan = ct_scan.astype(np.float32) - Config.NORMALIZATION_MIN
    return ct_scan

def world_to_voxel(coords, origin, spacing):
    """Convert world coordinates to voxel coordinates."""
    if origin is not None and spacing is not None:
        return np.round((coords - origin) / spacing).astype(int)
    return None

def preprocess_image(args):
    """Preprocess a single image patch."""
    ct_scan, coords, origin, spacing, output_size = args
    if ct_scan is None or origin is None or spacing is None:
        return None

    voxel_coords = world_to_voxel(coords, origin, spacing)
    if voxel_coords is None:
        return None

    z, y, x = voxel_coords
    z = np.clip(z, 0, ct_scan.shape[0] - 1)
    y = np.clip(y, 0, ct_scan.shape[1] - 1)
    x = np.clip(x, 0, ct_scan.shape[2] - 1)

    half_size = output_size[0] // 2
    y_start, y_end = np.clip([y - half_size, y + half_size], 0, ct_scan.shape[1]).astype(int)
    x_start, x_end = np.clip([x - half_size, x + half_size], 0, ct_scan.shape[2]).astype(int)

    cropped = ct_scan[z, y_start:y_end, x_start:x_end]
    if cropped.size == 0:
        return None

    resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_LINEAR)
    return resized

def augment_images(image, datagen, n_augmentations=Config.AUGMENTATIONS_PER_SAMPLE):
    """Apply data augmentation."""
    if image is None:
        return []
    image = image.reshape((1,) + image.shape + (1,))
    augmented = []
    for batch in datagen.flow(image, batch_size=1):
        augmented.append(batch[0, :, :, 0])
        if len(augmented) >= n_augmentations:
            break
    return augmented

def preprocess_luna16():
    """Main preprocessing function."""
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    
    # Load data
    annotations = pd.read_csv(Config.ANNOTATIONS_FILE)
    candidates = pd.read_csv(Config.CANDIDATES_FILE)
    
    # Balance classes
    positive_count = len(annotations)
    negative_candidates = candidates[candidates['class'] == 0].sample(n=positive_count, random_state=Config.SEED)
    positive_samples = annotations[['seriesuid', 'coordX', 'coordY', 'coordZ']].copy()
    positive_samples['class'] = 1
    all_samples = pd.concat([positive_samples, negative_candidates[['seriesuid', 'coordX', 'coordY', 'coordZ', 'class']]], ignore_index=True)
    
    # DataGen for augmentation
    datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.2, shear_range=0.2, horizontal_flip=True, fill_mode='nearest')
    
    images = []
    labels = []
    ct_cache = {}
    cache_limit = 5
    pool = Pool(processes=2)
    
    for seriesuid, group in tqdm(all_samples.groupby('seriesuid'), desc="Processing series"):
        mhd_files = glob.glob(os.path.join(Config.DATA_PATH, 'subset*/', f"{seriesuid}.mhd"))
        mhd_file = mhd_files[0] if mhd_files else None
        if not mhd_file:
            logger.warning(f"Skipping {seriesuid}: .mhd file not found")
            continue
        
        if seriesuid not in ct_cache:
            ct_scan, origin, spacing = load_mhd_file(mhd_file)
            if ct_scan is None:
                continue
            ct_scan = convert_to_hounsfield(ct_scan)
            ct_cache[seriesuid] = (ct_scan, origin, spacing)
        
        ct_scan, origin, spacing = ct_cache[seriesuid]
        tasks = [(ct_scan, np.array([row['coordX'], row['coordY'], row['coordZ']]), origin, spacing, Config.PATCH_SIZE) 
                 for _, row in group.iterrows()]
        
        results = pool.map(preprocess_image, tasks)
        for patch, (_, row) in zip(results, group.iterrows()):
            if patch is None:
                continue
            images.append(patch)
            labels.append(row['class'])
            
            # Augment
            augmented = augment_images(patch, datagen)
            images.extend(augmented)
            labels.extend([row['class']] * len(augmented))
        
        if len(ct_cache) >= cache_limit:
            ct_cache.clear()
            logger.info("Cleared CT cache")
    
    pool.close()
    pool.join()
    
    # Save processed data
    images = np.array(images)
    labels = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=Config.TRAIN_TEST_SPLIT, random_state=Config.SEED, stratify=labels)
    
    # Normalize
    X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min() + 1e-8)
    X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min() + 1e-8)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    size_str = f"{Config.PATCH_SIZE[0]}x{Config.PATCH_SIZE[1]}"
    np.save(os.path.join(Config.OUTPUT_PATH, f'X_train_{size_str}.npy'), X_train)
    np.save(os.path.join(Config.OUTPUT_PATH, f'X_test_{size_str}.npy'), X_test)
    np.save(os.path.join(Config.OUTPUT_PATH, f'y_train_{size_str}.npy'), y_train)
    np.save(os.path.join(Config.OUTPUT_PATH, f'y_test_{size_str}.npy'), y_test)
    
    logger.info(f"Saved data: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test

def build_model():
    """Build and compile CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(Config.PATCH_SIZE[0], Config.PATCH_SIZE[1], 1), padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss='binary_crossentropy',  # Optional: Replace with focal loss for imbalance
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

def train_and_evaluate():
    """Train model and evaluate."""
    X_train, X_test, y_train, y_test = preprocess_luna16() if not os.path.exists(os.path.join(Config.OUTPUT_PATH, 'X_train_128x128.npy')) else (
        np.load(os.path.join(Config.OUTPUT_PATH, 'X_train_128x128.npy')),
        np.load(os.path.join(Config.OUTPUT_PATH, 'X_test_128x128.npy')),
        np.load(os.path.join(Config.OUTPUT_PATH, 'y_train_128x128.npy')),
        np.load(os.path.join(Config.OUTPUT_PATH, 'y_test_128x128.npy'))
    )
    
    class_weights = compute_class_weight(Config.CLASS_WEIGHTS, classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    model = build_model()
    logger.info(model.summary())
    
    callbacks = [
        ModelCheckpoint(Config.MODEL_PATH, save_best_only=True, monitor='val_loss'),
        EarlyStopping(patience=Config.PATIENCE, monitor='val_loss', restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=5, monitor='val_loss')
    ]
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    history = model.fit(train_ds, validation_data=test_ds, epochs=Config.EPOCHS, class_weight=class_weight_dict, callbacks=callbacks, verbose=1)
    
    # Evaluation
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred, target_names=['Non-Nodule', 'Nodule']))
    
    # Visualizations
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.legend()
    plt.savefig(os.path.join(Config.OUTPUT_PATH, 'training_metrics.png'))
    plt.close()
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Nodule', 'Nodule'], yticklabels=['Non-Nodule', 'Nodule'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(Config.OUTPUT_PATH, 'confusion_matrix.png'))
    plt.close()
    
    # Sample predictions
    plt.figure(figsize=(15, 3))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(X_test[i, :, :, 0], cmap='gray')
        plt.title(f"True: {y_test[i]}\nPred: {y_pred[i]}")
        plt.axis('off')
    plt.savefig(os.path.join(Config.OUTPUT_PATH, 'sample_predictions.png'))
    plt.close()
    
    logger.info(f"Training complete. Results saved to {Config.OUTPUT_PATH}")

if __name__ == "__main__":
    train_and_evaluate()
