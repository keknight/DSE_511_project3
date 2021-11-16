# NASA: Asteroids Classification

Predicting whether an asteroid is hazardous or not using data provided by NASA.
The following machine learning algorithms will be used for this binary classification task:
- Naive bayes (NB)
- Support vector machine (SVM)
- Decision tree (DT)
- Random Forest (RF)

### Repository structure (in progress)
```
├── LICENSE
├── README.md               <- The project description and details.
├── .gitignore              <- Text file that tells Git which files or folders to ignore.
├── data
│   ├── raw                 <- The original data downloaded from https://www.kaggle.com/shrutimehta/nasa-asteroids-classification.
│   └── processed           <- The processed data used to train and evaluate the models.
│
├── models                  <- Trained and serialized models, model predictions, or model summaries
│
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── proposal.pdf        <- The project proposal.
│
├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
│                              generated with `pip freeze > requirements.txt`
│
├── src                     <- Source code for use in this project.
│   ├── preprocessing.py    <- The data preprocessing script.
│   └── models              <- Scripts to train and evaluate models.
|    └── train_nb_svc_model.py
|    └── predict_models.py
```


predict_model.py
│   │   └── train_model.py
### Code examples
```
## Load processsed data ##
# Load normalized data
with open("data/processed/train_scaled.pkl", "rb") as f:
    X_train_scaled, y_train = pkl.load(f)
    
with open("data/processed/test_scaled.pkl", "rb") as f:
    X_test_scaled, y_test = pkl.load(f)
    
    
# Load normalized + PCA data
with open("data/processed/train_scaled_pca.pkl", "rb") as f:
    X_train_scaled_pca, y_train = pkl.load(f)
    
with open("data/processed/test_scaled_pca.pkl", "rb") as f:
    X_test_scaled_pca, y_test = pkl.load(f)
```

### Acknowledgements

Dataset: All the data is from the (http://neo.jpl.nasa.gov/). This API is maintained by SpaceRocks Team: David Greenfield, Arezu Sarvestani, Jason English and Peter Baunach.
