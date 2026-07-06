import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D,
    Dense, UpSampling2D, Conv2DTranspose
)

def segnet(input_shape=(128, 128, 3)):
    segnet = Sequential([
        # Encoding layer
        Input(shape=input_shape),
        Conv2D(64, (3, 3), padding='same', activation='relu', strides=(1,1)),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(name='bn3'),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        Dense(1024, activation='relu'),
        Dense(1024, activation='relu'),

        # Decoding Layer
        UpSampling2D(),
        Conv2DTranspose(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2DTranspose(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2DTranspose(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),

        UpSampling2D(),
        Conv2DTranspose(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2DTranspose(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2DTranspose(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),

        UpSampling2D(),
        Conv2DTranspose(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2DTranspose(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2DTranspose(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),

        UpSampling2D(),
        Conv2DTranspose(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2DTranspose(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),

        UpSampling2D(),
        Conv2DTranspose(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2DTranspose(3, (3, 3), padding='same', activation='sigmoid'),
    ])
    return segnet


def get_segnet():
    return segnet()    
