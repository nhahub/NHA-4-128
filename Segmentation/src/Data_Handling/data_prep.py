# Functions to write:  download_kaggle_data(), merge_datasets(), split_train_test()
import glob
import os
from sklearn.model_selection import train_test_split

# Resized data
def load_resized_dataset():
  def load_resized_dataset(dataset_path, val_split=0.2, random_state=42):
    """ 
      Load the resized segmentation dataset.
      Returns: (train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks)
    """
    train_imgs = glob.glob(
        os.path.join(
            dataset_path,
            "train/input/img/*.jpg"
        )
    )

    train_masks = glob.glob(
        os.path.join(
            dataset_path,
            "train/ground truth/img/*.jpg"
        )
    )

    test_imgs = glob.glob(
        os.path.join(
            dataset_path,
            "test/input/img/*.jpg"
        )
    )

    test_masks = glob.glob(
        os.path.join(
            dataset_path,
            "test/ground truth/img/*.jpg"
        )
    )

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        train_imgs,
        train_masks,
        test_size=val_split,
        random_state=random_state
    )

    return (
        train_imgs,
        train_masks,
        val_imgs,
        val_masks,
        test_imgs,
        test_masks
    )

# ---------------- HAM data -----------------
import os
import glob
from sklearn.model_selection import train_test_split


def load_ham10000_dataset(dataset_path, train_size=0.7, val_size=0.15, random_state=42):
    """
    Load HAM10000 segmentation dataset.
    Returns: (train_ids ,val_ids ,test_ids)
    """

    raw_path = os.path.join(dataset_path, "data/raw")
    mask_path = os.path.join(dataset_path, "data/segmentation")

    images = glob.glob(os.path.join(raw_path, "*.jpg"))

    valid_ids = []

    for img in images:

        image_id = os.path.splitext(os.path.basename(img))[0]

        mask = os.path.join(
            mask_path,
            image_id + "_segmentation.png"
        )

        if os.path.exists(mask):
            valid_ids.append(image_id)

    train_ids, temp_ids = train_test_split(
        valid_ids,
        train_size=train_size,
        random_state=random_state
    )

    val_ratio = val_size / (1 - train_size)

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=1 - val_ratio,
        random_state=random_state
    )

    return train_ids, val_ids, test_ids
