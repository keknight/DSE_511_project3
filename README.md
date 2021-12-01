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
To predict whether an asteroid is hazardous or not, four supervised machine algorithms were trained on data provided by the NASA API called NeoWS (Near Earth Object Web Service) and which is readily available on Kaggle (https://www.kaggle.com/shrutimehta/nasa-asteroids-classification). Specifically, the following four algorithms were used to perform this binary classification problem:
- Naive bayes (NB)
- Support vector machine (SVM)
- Decision tree (DT)
- Random Forest (RF)

Prior to modeling, the data was split into a train and test set (80:20 ratio). In addition, min-max normalizion and principal component analysis (PCA) for dimensionality reduction was applied. Then each model was trained on both the normalized and normalized+pca train data, and evaluated on the normalized and normalized+pca test data. Lastly, 

### Repository structure (in progress)
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

### Code examples
```
# To install project dependencies
pip install -r requirements.txt

# The following command will perform data processing, model training and model evaluation
python main.py
```



### Acknowledgements

Dataset: All the data is from the (http://neo.jpl.nasa.gov/). This API is maintained by SpaceRocks Team: David Greenfield, Arezu Sarvestani, Jason English and Peter Baunach.
