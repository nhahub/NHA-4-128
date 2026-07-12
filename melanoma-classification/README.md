# Melanoma Classification

A production-grade deep learning pipeline for classifying melanoma skin lesions from dermoscopic images using Convolutional Neural Networks (CNNs). The project supports three model architectures and automatically downloads the dataset from Kaggle when run locally.

## Features

- **3 model architectures**: Vanilla CNN, Custom CNN, and MobileNetV2 (transfer learning)
- **Automatic dataset download** from Kaggle via `kagglehub` — no manual data setup needed
- **Data augmentation** for improved generalization
- **Callbacks**: model checkpointing, early stopping, and learning rate reduction
- **Comprehensive evaluation**: accuracy, classification report, and confusion matrix
- **Training history visualization** (loss and accuracy curves)
- **Runs on Kaggle notebooks** and **local machines** seamlessly

## Dataset

The project uses the **SIIM-ISIC Melanoma Classification** dataset, resized to 256×256 JPEG images:

- **Source**: [cdeotte/jpeg-melanoma-256x256](https://www.kaggle.com/datasets/cdeotte/jpeg-melanoma-256x256)
- **Size**: ~33,126 training images (584 malignant, ~1.8%)
- **Classes**: binary — *benign* (0) and *malignant* (1)
- **Imbalance**: heavily imbalanced; the custom CNN uses augmentation to mitigate this

When you run the pipeline outside Kaggle, `kagglehub` automatically downloads the dataset on first run (no API key required for public datasets).

## Project Structure

```
melanoma-classification/
├── data/                          # Auto-populated when downloading from Kaggle
├── src/
│   ├── __init__.py
│   ├── config.py                  # Paths, hyperparameters, dataset ID
│   ├── data_loader.py             # Kaggle download + data generators
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vanilla_cnn.py         # Simple 2-conv CNN baseline
│   │   ├── custom_cnn.py          # Deeper 3-conv CNN with augmentation
│   │   └── mobilenet_model.py     # MobileNetV2 transfer learning
│   ├── training.py                # Callbacks and fit loop
│   ├── evaluation.py              # Accuracy, classification report, confusion matrix
│   └── utils.py                   # Loss/accuracy plotting
├── main.py                        # Entry point — runs the full pipeline
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/nhahub/NHA-4-128.git
cd NHA-4-128/melanoma-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
python main.py
```

On first run, `kagglehub` downloads the ~1.5 GB dataset automatically. Subsequent runs use the cached copy.

To run on **Kaggle notebooks**, place the dataset at `/kaggle/input/melanoma-256x256/` — the code detects the Kaggle environment and uses the pre-loaded data.

## Model Architectures

| Model | Layers | Input Size | Parameters | Notes |
|-------|--------|------------|------------|-------|
| **Vanilla CNN** | 2 Conv + 2 MaxPool + Dense | 150×150×3 | ~1.2M | Baseline — no augmentation |
| **Custom CNN** | 3 Conv + 3 MaxPool + Dense | 150×150×3 | ~2.8M | Trained with augmentation, 50 epochs |
| **MobileNetV2** | Pretrained base + Dense | 224×224×3 | ~2.3M | Transfer learning, 20 epochs |

## Results

The pipeline trains all three models sequentially and outputs:

- Loss/accuracy curves for Custom CNN and MobileNetV2
- Confusion matrix and classification report for MobileNetV2
- Predictions on the validation set using the Custom CNN

## Requirements

- Python 3.8+
- TensorFlow 2.x
- kagglehub
- See `requirements.txt` for full list

## License

MIT
