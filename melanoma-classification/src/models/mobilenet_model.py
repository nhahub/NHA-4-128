from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import MobileNetV2
from src.config import IMG_SHAPE_MOBILENET

def build_mobilenet_model(train_generator, img_shape=IMG_SHAPE_MOBILENET):
    # Preserving exact variable assignment from notebook
    class_count = len(list(train_generator.class_indices.keys()))
    
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')
    base_model.trainable = False

    model_3 = Sequential([
        base_model,
        Dense(1, activation='sigmoid')
    ])
    model_3.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model_3