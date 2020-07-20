auto-portrait-segmentation
==============================

Apply an automatic portrait segmentation model (aka image matting) to <a href="https://github.com/switchablenorms/CelebAMask-HQ">celebrity face dataset</a>.

## Download Data
To download dataset visit above link or <br />
`cd src/data` <br />
`python make_dataset.py`

## Get pretrained weights
`./models/get_pretrained.sh` <br /> or <br />
Google Drive Link : https://drive.google.com/drive/folders/1cF8D21yGyqogenu3DCY5fZLpAnuYvtlq?usp=sharing

## Model
<a href="https://arxiv.org/pdf/1706.05587.pdf">DeepLabv3</a> model has been used here, with input of `224x224x3` and output `224x224x1` mask.

## Notebooks
Notebooks folder contains a basic and in-depth EDA notebooks, as well as model inspection(`model_select.ipynb`). <br />
`notebooks/demo.ipynb` -> contains an implementation of the pretrained model.<br />
`notebooks/model_performance.ipynb` -> Gives an overview of the model training loss and validation metrics stats.

## Training Model
Run main.py script.<br />
To get argument details : `python src/main.py -h`

## Performance Metrics
### Training Loss
![Loss Plot](https://github.com/rahatsantosh/autoportrait_seg/blob/master/reports/fig/loss_plot.png)
### F1 Score
![F1 Metric Plot](https://github.com/rahatsantosh/autoportrait_seg/blob/master/reports/fig/f1_metric.png)
### Area Under ROC Curve
![AUROC Metric Plot](https://github.com/rahatsantosh/autoportrait_seg/blob/master/reports/fig/auroc_metric.png)

Project Organization
------------

    ├── LICENSE
    ├── README.md                               <- The top-level README for developers using this project.
    ├── data
    │   ├── processed                           <- The final, canonical data sets for modeling.
    │   ├── interim                             <- Partially processed data.
    │   └── raw                                 <- The original, immutable data dump.
    │
    ├── models                                  <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                               <- Jupyter notebooks.
    |   |
    |   ├── raw_data_eda.ipynb                  <- Basic EDA, on raw dataset size, image size, format etc.
    |   |
    |   ├── advanced_eda.ipynb                  <- Advanced EDA performed on masks and mask averages.
    |   |
    |   ├── model_select.ipynb                  <- Possible model architectures and known metrics.
    |   |
    |   ├── model_performance.ipynb             <- Performance metrics of refitted, pretrained model.
    |   |
    |   └── demo.ipynb                          <- Implementation of pretrained model, post training, on sample image.
    │
    ├── references                              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt                        <- The requirements file for reproducing the analysis environment
    │
    ├── src           				          <- Source code for use in this project.
    │   │
    │   ├── data                                <- Scripts to download or generate data
    │   │
    │   ├── utils                                <- Scripts utilities used during data generation or training
    │   │
    │   └── training                            <- Scripts to train models

## Acknowledgements
* Model architecture based on <a href="https://arxiv.org/pdf/1706.05587.pdf">DeepLabv3</a>
* Dataset from https://github.com/switchablenorms/CelebAMask-HQ
* This project repo was based on cookiecutter template -> https://github.com/tdeboissiere/cookiecutter-deeplearning
* `src/training/trainer.py` module based on https://github.com/msminhas93/DeepLabv3FineTuning/blob/master/trainer.py

### MIT License
All code in the repo is under MIT license, external code and data are under respective licenses.<br />
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
`[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)`
