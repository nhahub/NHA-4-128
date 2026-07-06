DATASETs from Kaggle

Data 1 : Skin Cancer HAM10000: Raw and Lesion Segmentation
https://www.kaggle.com/datasets/mfazrinizar/skin-cancer-ham10000-raw-and-lesion-segmentation

Data 2 : Resized Dataset for Skin Lesion Segmentation-new
https://www.kaggle.com/datasets/apurboshahidshawon/resized-dataset-for-skin-lesion-segmentation-new

* The architecture we are implementing is suggested in [this](https://arxiv.org/pdf/1511.00561.pdf) paper.

```
project arch/
│
├── readme.md                  
├── main.py                    # Entry point
├── requiretments.txt          
│
├── src/
    ├── Data Handling/
    │   ├── data_prep.py           # Load datasets, processing and augmentation
    │   ├── merger.py              # Merge datasets
    │   └── pipeline.py            # tf.data pipelines
    │
    ├─  models/
    │   ├── unet.py                # U-Net architecture
    │   └── segnet.py              # Seg_Net architecture
    │
├── train.py             # model.fit()
├── config.py            # All constants & hyperparameters
├── metrices.py          # accuracy and loss
└── visuals.py           # plotting the results
│
│
└── notebooks/
    └── unet-segmentation.ipynb
```
    
