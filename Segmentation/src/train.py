from model import get_model
from config import EPOCHS
from metrices import dice_coef, dice_loss
import tensorflow as tf

def get_callbacks(model_name):
  checkpoint = tf.keras.callbacks.ModelCheckpoint(
      "best_segnet_model.keras",
      monitor='val_accuracy',
      verbose=1,
      save_best_only=True,
      mode='max'
  )

  early_stopping = tf.keras.callbacks.EarlyStopping(
      monitor='val_loss',
      patience=15, 
      restore_best_weights=True, 
      verbose=1
  )

  reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
      monitor='val_loss',
      patience=5,
      factor=0.2,
      min_lr=1e-6,
      verbose=1
  )
  return [early_stopping, reduce_lr, checkpoint]



def train_model(model, train_ds, val_ds):

    model.compile(
        optimizer="adam",
        loss=dice_loss,
        metrics=["accuracy", dice_coef]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=get_callbacks
    )

    model.save("segmentation_model.keras")

    return history

