# NASA: Asteroids Classification

Predicting whether an asteroid is hazardous or not using data provided by NASA.
The following machine learning algorithms will be used for this binary classification problem:
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
├── notebooks               <- Jupyter notebooks for data analysis and visualization.
│
├── models                  <- Trained and serialized models, model predictions, or model summaries.
│
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── proposal.pdf        <- The project proposal.
│
├── requirements.txt        <- The requirements file for reproducing the analysis environment. Contains project dependencies.
│
├── src                     <- Source code for use in this project.
│   ├── preprocessing.py    <- The data preprocessing script.
│   └── models              <- Scripts to train and evaluate models.
│    ├── train_models.py
│    └── predict_models.py
```

```
### Code examples

# To install project dependencies
pip install -r requirements.txt

```



### Acknowledgements

Dataset: All the data is from the (http://neo.jpl.nasa.gov/). This API is maintained by SpaceRocks Team: David Greenfield, Arezu Sarvestani, Jason English and Peter Baunach.
