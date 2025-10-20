import os

class Config:
    # Paths
    DATA_PATH = '/content/drive/MyDrive/LUNA16/'  # Adjust for local/Colab
    ANNOTATIONS_FILE = os.path.join(DATA_PATH, 'annotations.csv')
    CANDIDATES_FILE = os.path.join(DATA_PATH, 'candidates.csv')
    OUTPUT_PATH = os.path.join(DATA_PATH, 'processed_1/')
    MODEL_PATH = os.path.join(OUTPUT_PATH, 'best_nodule_model.h5')
    
    # Preprocessing
    PATCH_SIZE = (128, 128)
    AUGMENTATIONS_PER_SAMPLE = 2
    TRAIN_TEST_SPLIT = 0.2
    BATCH_SIZE = 32
    NORMALIZATION_MIN = -1024  # Hounsfield window
    
    # Model
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    CLASS_WEIGHTS = 'balanced'  # Or dict if custom
    PATIENCE = 10
    
    # Reproducibility
    SEED = 42
