from Data_Handling.merger import merge_datasets
from Data_Handling.pipeline import create_train_dataset, create_validation_dataset, create_test_dataset
from Segmentation.src.config import SEED
from config import RESIZED_DATA_PATH, HAM_DATA_PATH, image_size, batch_size, seed
from Data_Handling.data_prep import load_resized_dataset, load_ham10000_dataset
from model import build_unet
from train import train_model

def main():
    resized_dataset = load_resized_dataset(
        dataset_path=RESIZED_DATA_PATH,
        val_split=0.2,
        random_state=SEED,
    )
    ham_dataset = load_ham10000_dataset(
        dataset_path=HAM_DATA_PATH,
        random_state=SEED,
    )
    print("Datasets loaded successfully.\n")
    print("=" * 60)
    print("Merging datasets...")
    print("=" * 60)

    (
        train_images,
        train_masks,
        val_images,
        val_masks,
        test_images,
        test_masks,
    ) = merge_datasets(
        resized_dataset,
        ham_dataset,
        HAM_DATA_PATH,
    )

    print("Merged successfully.\n")

    print("=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    print(f"Training Images   : {len(train_images)}")
    print(f"Validation Images : {len(val_images)}")
    print(f"Testing Images    : {len(test_images)}")

    print()

    print("=" * 60)
    print("Building tf.data pipelines...")
    print("=" * 60)

    train_ds, val_ds, test_ds = create_train_dataset(
        train_images,
        train_masks,
        image_size=image_size,
        batch_size=batch_size,
    ), create_validation_dataset(
        val_images,
        val_masks,
        image_size=image_size,
        batch_size=batch_size,
    ), create_test_dataset(
        test_images,
        test_masks,
        image_size=image_size,
        batch_size=batch_size,
    )
    print("tf.data pipelines created successfully.\n")

    # Model Construction (next module)
    model = build_unet(image_size=image_size)
    history = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=10,
    )
    return history

if __name__ == "__main__":

    train_ds, val_ds, test_ds = main()
