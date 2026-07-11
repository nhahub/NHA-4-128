import os
import kagglehub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import TRAIN_PATH, BATCH_SIZE, TARGET_SIZE


def download_from_kaggle():
    """Download the melanoma dataset from Kaggle using kagglehub."""
    dataset_id = "cdeotte/jpeg-melanoma-256x256"
    print(f"Downloading dataset from Kaggle ({dataset_id})...")
    download_path = kagglehub.dataset_download(dataset_id)
    print(f"Dataset downloaded to: {download_path}")
    return download_path


def _resolve_train_path(train_path):
    if train_path is not None:
        return train_path
    if os.path.exists(TRAIN_PATH):
        return TRAIN_PATH
    download_path = download_from_kaggle()
    candidate = os.path.join(download_path, "train")
    if os.path.isdir(candidate):
        return candidate
    return download_path


def get_data_generators(train_path=None, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, augment=False):
    resolved_path = _resolve_train_path(train_path)

    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )

    train_generator = train_datagen.flow_from_directory(
        resolved_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        resolved_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator
