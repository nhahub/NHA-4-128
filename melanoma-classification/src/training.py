import tensorflow as tf

def get_callbacks(model_name):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f'best_{model_name}_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    return [checkpoint, early_stopping, reduce_lr]

def train_model(model, train_generator, val_generator, epochs, callbacks):
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )
    return history