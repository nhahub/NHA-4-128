# Melanoma Classification Project

A production-grade Python project for classifying melanoma images using Convolutional Neural Networks (CNNs). This project is a direct structural extraction from a fully tested Jupyter Notebook, preserving 100% of the original logic, model architectures, and execution flow.

##  Folder Structure
```text
melanoma-classification/
├── data/                      # (Optional) Store data here or use a symlink to Kaggle data
├── notebooks/                 # Keep your original .ipynb here for reference/EDA
├── src/                       # Source code
│   ├── config.py              # Paths, hyperparameters, and constants
│   ├── data_loader.py         # Data generators and preprocessing
│   ├── models/                # Model architectures (Vanilla, Custom, MobileNetV2)
│   ├── training.py            # Callbacks and training loops
│   ├── evaluation.py          # Prediction, Confusion Matrix, Reports
│   └── utils.py               # Visualization helpers
├── main.py                    # Entry point to run the pipeline
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation