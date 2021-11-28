# NASA: Asteroids Classification

Predicting whether an asteroid is hazardous or not using data provided by NASA.
The following machine learning algorithms will be used for this binary classification problem:
- Naive bayes (NB)
- Support vector machine (SVM)
- Decision tree (DT)
- Random Forest (RF)

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
