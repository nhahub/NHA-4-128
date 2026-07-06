# processing and augmentation + dataset managing(train,val,test)
import tensorflow as tf
from config import IMG_SIZE, BATCH_SIZE
from merger import train_img_paths, train_mask_paths, val_img_paths, val_mask_paths, test_img_paths, test_mask_paths

def load_and_preprocess(img_path, mask_path):
    img = tf.io.read_file(img_path) 
    img = tf.image.decode_jpeg(img, channels=3) 
    img = tf.image.resize(img, IMG_SIZE) 
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path) 
    mask = tf.image.decode_image(mask, channels=3, expand_animations=False) 
    mask = tf.image.resize(mask, IMG_SIZE) 
    mask = tf.cast(mask, tf.float32) / 255.0 
    mask = tf.where(mask > 0.5, 1.0, 0.0) 
    return img, mask


def apply_augmentation(img, mask):
    img = tf.image.random_flip_left_right(img)
    mask = tf.image.random_flip_left_right(mask)
    
    img = tf.image.random_flip_up_down(img)
    mask = tf.image.random_flip_up_down(mask)
    
    img = tf.image.random_brightness(img, max_delta=0.1)
    return img, mask


def create_train_dataset():
  train_ds = tf.data.Dataset.from_tensor_slices((train_img_paths, train_mask_paths))
  train_ds = train_ds.shuffle(buffer_size=len(train_img_paths))

  # PREPROCESS and AUGMENT
  train_ds = train_ds.map(load_and_preprocess , num_parallel_calls=tf.data.AUTOTUNE)    
  train_ds = train_ds.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
  train_ds = train_ds.batch(BATCH_SIZE)
  print("Training Pipeline is ready with Shuffling and Augmentation!")
  return train_ds

def create_validation_dataset():
  val_ds = tf.data.Dataset.from_tensor_slices((val_img_paths, val_mask_paths))
  val_ds = val_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUN)
  val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
  return val_ds

def create_test_dataset():
  # Create Testing Dataset
  test_ds = tf.data.Dataset.from_tensor_slices((test_img_paths, test_mask_paths))
  test_ds = test_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
  test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
  return test_ds

