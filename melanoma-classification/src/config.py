import os

# Kaggle dataset identifier (used by kagglehub for local downloads)
KAGGLE_DATASET = "cdeotte/jpeg-melanoma-256x256"

# Paths (Kaggle notebook paths — used when running on Kaggle)
TRAIN_PATH = '/kaggle/input/melanoma-256x256/train'
TEST_PATH = '/kaggle/input/melanoma-256x256/test'
SAVE_DIR = '/kaggle/working/'

# Local data directory (used when running outside Kaggle)
LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Hyperparameters
BATCH_SIZE = 32
TARGET_SIZE = (150, 150)
IMG_SHAPE_MOBILENET = (224, 224, 3)
EPOCHS_MODEL_2 = 50
EPOCHS_MODEL_3 = 20
