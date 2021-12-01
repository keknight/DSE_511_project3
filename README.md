# NASA Asteroid Classification

## Project Intro/Objective
The purpose of this project is to predict whether an asteroid is hazardous or not.

## Methods Used
* Data analysis and visualization
* Dimensionality reduction
* Machine Learning
* Predictive Modeling
* Feature importance

## Project Description
To predict whether an asteroid is hazardous or not, four supervised machine algorithms are trained on data provided by the NASA API called NeoWS (Near Earth Object Web Service) and which is readily available on Kaggle (https://www.kaggle.com/shrutimehta/nasa-asteroids-classification). Specifically, the following four algorithms are used to perform this binary classification problem:
- Naive bayes (NB)
- Support vector machine (SVM)
- Decision tree (DT)
- Random Forest (RF)

Prior to modeling, the data is split into a train and test set (80:20 ratio). In addition, min-max normalizion and principal component analysis (PCA) for dimensionality reduction are applied. Then each algorithm is built on both the normalized and normalized+pca train data (i.e., a total of eight models), and evaluated on the normalized and normalized+pca test data, respectively. Lastly, feature importance is applied to identify the most important input features.

## Repository structure (in progress)
```
├── data
│   ├── raw                   <- The original data downloaded from https://www.kaggle.com/shrutimehta/nasa-asteroids-classification.
│   └── processed             <- The processed data used to train and evaluate the models.
│
├── reports                   <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures               <- Generated figures to be used in reporting.
│   └── proposal.pdf          <- The project proposal.
│
├── notebooks                 <- Jupyter notebooks for data analysis and visualization.
│   └── data_analysis.ipynb
│
├── models                    <- Trained and serialized models, model predictions, or model summaries.
│
├── src                       <- Source code for use in this project.
│   ├── models                <- Scripts to train and evaluate models.
│   │   ├── train_models.py
│   │   └── predict_models.py
│   │
│   ├── data_preprocessing.py
│   │
│   ├── feature_importance.py
│   │
│   ├── main.py
│   │
│   ├── utils.py
│   │
│   └── visualization_class_results.py
│ 
├── .gitignore                <- Text file that tells Git which files or folders to ignore.
│  
├── LICENSE         
│  
├── README.md                 <- The project description and details.
│  
└── requirements.txt          <- The requirements file for reproducing the analysis environment. Contains project dependencies.
```

## Getting Started
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).

2. Raw data can be found [here](https://github.com/keknight/DSE_511_project3/tree/main/data/raw) within this repo, or can be downloaded [here](https://towardsdatascience.com/nasa-asteroid-classification-6949bda3b1da). And the processed data can be found [here](https://github.com/keknight/DSE_511_project3/tree/main/data/processed). 

4. Data analysis/visualization notebooks are being kept [here](https://github.com/keknight/DSE_511_project3/tree/main/notebooks), and data preprocessing, model training/predicting, and feature importance scripts are being kept [here](https://github.com/keknight/DSE_511_project3/tree/main/src). 

5. Reports and figures can be found [here](https://github.com/keknight/DSE_511_project3/tree/main/reports).

## Code examples
```
# To install project dependencies
pip install -r requirements.txt

# The following command will perform data processing, model training and model evaluation
python main.py
```

## Project Members

|Name     |  Slack Handle   | 
|---------|-----------------|
|[Katie Knight](https://github.com/keknight) |     @keknight    |
|[Anna-Maria Nau](https://github.com/annamarianau)| @annamarianau        |
|[Christoph Metzner](https://github.com/cmetzner93) |     @cmetzner93    |

### Acknowledgements

Dataset: All the data is from the (http://neo.jpl.nasa.gov/). This API is maintained by SpaceRocks Team: David Greenfield, Arezu Sarvestani, Jason English and Peter Baunach.
