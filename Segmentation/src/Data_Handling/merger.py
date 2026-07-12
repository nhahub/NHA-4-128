import os


def merge_datasets(resized_data, ham_ids, ham_dataset_path,):
    """
    Merge the resized dataset with HAM10000.
    Returns: (train_img_paths, train_mask_paths, val_img_paths, val_mask_paths, test_img_paths, test_mask_paths)
    """
    (
      resized_train_imgs,
      resized_train_masks,
      resized_val_imgs,
      resized_val_masks,
      resized_test_imgs,
      resized_test_masks,
    ) = resized_data

    train_ids, val_ids, test_ids = ham_ids

    raw_dir = os.path.join(ham_dataset_path, "data/raw")
    mask_dir = os.path.join(ham_dataset_path, "data/segmentation")

    train_img_paths = (
        resized_train_imgs + 
        [ os.path.join(raw_dir, img + ".jpg") for img in train_ids]
    )

    train_mask_paths = (
        resized_train_masks + 
        [os.path.join(mask_dir, img + "_segmentation.png") for img in train_ids]
    )

    val_img_paths = (
        resized_val_imgs+
        [os.path.join(raw_dir, img + ".jpg") for img in val_ids]
    )

    val_mask_paths = (
        resized_val_masks + 
        [os.path.join(mask_dir, img + "_segmentation.png") for img in val_ids]
    )

    test_img_paths = (
        resized_test_imgs + 
        [ os.path.join(raw_dir, img + ".jpg") for img in test_ids]
    )

    test_mask_paths = (
        resized_test_masks + 
        [ os.path.join(mask_dir, img + "_segmentation.png") for img in test_ids]
    )

    return (
        train_img_paths,
        train_mask_paths,
        val_img_paths,
        val_mask_paths,
        test_img_paths,
        test_mask_paths,
    )