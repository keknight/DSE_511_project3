"""
Title: Binary Classification - Naive Bayes and Support Vector Machine
Author: Christoph Metzner

This file contains source code to perform a binary classification using the machine learning algorithms naive bayes
and support vector machine. The classification is performed on normalized data and data with reduced dimensions using
principal component analysis.

"""

import os
import numpy as np
# Type hinting
from typing import List
from typing import Dict
from typing import Tuple
from typing import Union
from typing import TypeVar
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
PandasSeries = TypeVar('pandas.core.series.Series')
SklearnClassifier = TypeVar('sklearn.svm._classes.SVC')

# Hypertuning GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

# ML-Classification Algorithms
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# use pickle to save trained model
import pickle


# Set global variable seed for reproducibility
seed = 0


def init_classifier(clf_v: str) -> Tuple[SklearnClassifier, Dict[str, int]]:
    """
    Function that returns the classifier and parameter grid for hyper-parameter tuning.

    Parameters
    ----------
    clf: str
        string variable indicating which SKlearnClassifier should be initialized with hyperparameter space for grid
        search

    Returns
    -------
    classifier: SklearnClassifier
        Sklearn classifier object
    param_grid: Dict[str, int, float]
        Dictionary containing hyper-parameters that should be investigated during gridsearch cv
    """

    if clf_v == 'nb':
        clf = GaussianNB()
        param_grid = {'var_smoothing': np.logspace(0, -9, num=100)}

    elif clf_v == 'svc':
        clf = SVC(random_state=seed)
        param_grid = [
          {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
          {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
         ]

    return clf, param_grid


def train_model(
        X_train: PandasDataFrame,
        y_train: PandasSeries,
        clf_v: str,
        cv_m: int,
        pca: bool=False) -> List[Union[float, List[float]]]:
    """
    This function performs hyperparameter tuning via the function GridSearchCV, takes the best estimator of the
    respective classifier for model evaluation. The splits during cross-validation are stratified.

    Parameters
    ----------
    X_train: PandasDataFrame
        Dataframe containing only the cleaned, preprocessed training data (tweets).
    X_test: PandasDataFrame
        Dataframe containing only the cleaned, preprocessed testing data (tweets).
    y_train: PandasSeries
        Series that contains the respective ground-truth labels for the training data.
    y_test: PandasSeries
        Series that contains the repsective ground-truth labels for the testing data.
    clf_v: str
        Variable that indicates the type of initialized classifier ('nb', 'svc')
    cv_m: int
        Variable that holds number of stratified splits (folds). Stratified splits have similar label distribution.
    pca: bool
        Variable indicating whether scaled or scaled + pca data is used

    Returns
    -------
    SklearnClassifier
        Best model after hyperparameter-tuning
    """
    # initialize classifier
    clf, param_grid = init_classifier(clf_v=clf_v)

    # perform hyper-parameter tuning
    cv_m = StratifiedKFold(cv_m)
    grid_clf = GridSearchCV(clf, param_grid, cv=cv_m, scoring='f1', n_jobs=-1)
    grid_clf.fit(X_train, y_train)

    # get best model after hyper-parameter tuning
    clf_best = grid_clf.best_estimator_
    # train best model
    clf_best.fit(X_train, y_train)
    if pca:
        pca_s = '_pca'
    else:
        pca_s = ''
    with open(os.path.join('..', 'models', f'trained_{clf_v}{pca_s}_model.pkl'), 'wb') as f:
        pickle.dump(clf_best, f)

    return clf_best
