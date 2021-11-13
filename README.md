# DSE 511 - Introduction to Data Science and Computing I
# NASA: Asteroids Classification

Project description:
This github repository is for the binary classification of the impact risk of near-earth objects (NEOs) for planet earth and the earth's population. The main research question is whether standard "off-the-shelf" algorithms can produce outstanding classification performance. The following algorithms will be trained to predict whether an asteroid is hazardous or not:
- Naive bayes (NB)
- Support vector machine (SVM)
- Decision tree (DT)
- XGBoost

Repository structure (in progress):
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
```
