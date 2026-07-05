from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import TRAIN_PATH, BATCH_SIZE, TARGET_SIZE

def get_data_generators(train_path=TRAIN_PATH, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, augment=False):
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
        train_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator