"""
Title: Binary Classification - Naive Bayes and Support Vector Machine
Author: Christoph Metzner

This file contains source code to perform a binary classification using the machine learning algorithms naive bayes
and support vector machine. The classification is performed on normalized data and data with reduced dimensions using
principal component analysis.

"""

import numpy as np
import pandas as pd
import random
from typing import List
from typing import Dict
from typing import Tuple
from typing import Union
from typing import TypeVar
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
PandasSeries = TypeVar('pandas.core.series.Series')
SklearnClassifier = TypeVar('sklearn.svm._classes.SVC')

# Libraries

# Hypertuning GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

# ML-Classification Algorithms
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Performance Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


seed = 0

def ml_classification(
        X_train: PandasDataFrame,
        X_test: PandasDataFrame,
        y_train: PandasSeries,
        y_test: PandasSeries,
        classifier: SklearnClassifier,
        param_grid: Dict[str, Union[int, float, str]],
        cv_m: int) -> List[Union[float, List[float]]]:
    """This function performs hyperparameter tuning via the function GridSearchCV, takes the best estimator of the respective
    classifier for model evaluation. The splits during cross-validation are stratified.


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
    classifier: SklearnClassifier
        Variable that holds the respective initialized classifier.
    param_grid: dict
        A python dictionary that contains all parameters with their values for hyperparameter tuning.
    cv_m: int
        Variable that holds number of stratified splits (folds).

    Returns
    -------
    result_classifier: List
        A list containing all the performance metrics of the model.
    """
    grid_clf = GridSearchCV(classifier, param_grid, cv=cv_m, scoring='f1', n_jobs=-1)
    grid_clf.fit(X_train, y_train)

    clf_best = grid_clf.best_estimator_
    clf_best.fit(X_train, y_train)

    y_pred = clf_best.predict(X_test)
    # y_pred_proba = clf_best.predict_proba(X_test)

    precision = precision_score(y_test, y_pred, average='binary', pos_label=1)
    recall = recall_score(y_test, y_pred, average='binary', pos_label=1)
    f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)

    print(f'\nModel Performance for {clf_best}:')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-Score: {f1}')

    # Plot a confusion matrix of the results
    print("\n--- Confusion matrix for test data ---")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    tp = conf_matrix[0][0]
    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]
    tn = conf_matrix[1][1]
    result = [clf_best, precision, recall, f1, tp, fp, fn, tn]

    return result

