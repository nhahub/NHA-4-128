import matplotlib.pyplot as plt
import numpy as np


def show_dataset_samples(dataset, num_samples=2):
    """
    Display image-mask pairs from a tf.data.Dataset.

    Args:
        dataset: tf.data.Dataset yielding (images, masks)
        num_samples: Number of samples to display
    """

    for images, masks in dataset.take(1):

        fig, axes = plt.subplots(
            num_samples,
            2,
            figsize=(8, 4 * num_samples)
        )

        if num_samples == 1:
            axes = np.expand_dims(axes, axis=0)

        for i in range(num_samples):

            # Image
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title(f"Image {i+1}")
            axes[i, 0].axis("off")

            # Mask
            axes[i, 1].imshow(
                masks[i].numpy().squeeze(),
                cmap="gray"
            )
            axes[i, 1].set_title(f"Mask {i+1}")
            axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()

# ---------------------------------------

def plot_predictions(model, dataset, num_samples=3, threshold=0.5):
    """
    Plot predictions on test images.

    Args:
        model: Trained segmentation model
        dataset: tf.data.Dataset
        num_samples: Number of images to visualize
        threshold: Binary threshold
    """

    predictions = model.predict(dataset, verbose=0)

    sample_idx = 0

    for images, masks in dataset:

        for i in range(images.shape[0]):

            if sample_idx >= num_samples:
                return

            image = images[i].numpy()
            gt_mask = masks[i].numpy()

            pred_mask = predictions[sample_idx]
            pred_mask = (pred_mask > threshold).astype(np.float32)

            overlay = image.copy()
            overlay[pred_mask.squeeze() == 0] = 0

            fig, axes = plt.subplots(
                1,
                4,
                figsize=(18, 5)
            )

            axes[0].imshow(image)
            axes[0].set_title("Input Image")
            axes[0].axis("off")

            axes[1].imshow(
                gt_mask.squeeze(),
                cmap="gray"
            )
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(
                pred_mask.squeeze(),
                cmap="gray"
            )
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            axes[3].imshow(overlay)
            axes[3].set_title("Overlay")
            axes[3].axis("off")

            plt.tight_layout()
            plt.show()

            sample_idx += 1        