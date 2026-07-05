import os

# Paths (Preserving exact Kaggle notebook paths)
TRAIN_PATH = '/kaggle/input/melanoma-256x256/train'
TEST_PATH = '/kaggle/input/melanoma-256x256/test'
SAVE_DIR = '/kaggle/working/'

# Hyperparameters
BATCH_SIZE = 32
TARGET_SIZE = (150, 150)
IMG_SHAPE_MOBILENET = (224, 224, 3)
EPOCHS_MODEL_2 = 50
EPOCHS_MODEL_3 = 20