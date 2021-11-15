"""
Title: Binary Classification - Naive Bayes and Support Vector Machine
Author: Christoph Metzner

This file contains source code to perform a binary classification using the machine learning algorithms naive bayes
and support vector machine. The classification is performed on normalized data and data with reduced dimensions using
principal component analysis.

"""
import pandas as pd

import pickle
import os

# Performance Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from typing import TypeVar
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
PandasSeries = TypeVar('pandas.core.series.Series')
SklearnClassifier = TypeVar('sklearn.svm._classes.SVC')


def predict_model(
        X_test: PandasDataFrame,
        y_test: PandasSeries,
        clf_v: str,
        pca: bool=False) -> PandasDataFrame:
    """
    Function that does model evaluation on unseen testing dataset.

    Parameters
    ----------
    X_test: PandasDataFrame
        Dataframe that contains all feature space of test data
    y_test: PandasSeries
        Series that contains the ground-truth labels of the test data
    clf_v: str
        Variable that indicates the type of initialized classifier ('nb', 'svc')
    pca: bool
        Boolean variable indicating if pca data used or not.

    Returns
    -------
    PandasDataFrame
        A dataframe containing the performance metric results
    """
    if pca:
        pca_s = '_pca'
    else:
        pca_s = ''
    # loading the best model from hyper-parameter tuning and training
    with open(os.path.join('..', 'models', f'trained_{clf_v}{pca_s}_model.pkl'), 'rb') as f:
        clf = pickle.load(f)

    y_pred = clf.predict(X_test)  # get prediction for test dataset

    # Do model performance evaluation
    f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)
    precision = precision_score(y_test, y_pred, average='binary', pos_label=1)
    recall = recall_score(y_test, y_pred, average='binary', pos_label=1)

    print(f'\nModel Performance for {clf}:')
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

    results = [[clf_v, pca, f1, precision, recall, tp, fp, fn, tn]]
    df_results = pd.DataFrame(
        results,
        columns=['model',
                 'pca',
                 'f1_score',
                 'precision',
                 'recall',
                 'tp',
                 'fp',
                 'fn',
                 'tn'])

    save_results(df_results=df_results)

    return df_results


def save_results(df_results: PandasDataFrame):
    """
    Function that saves the results to a csv file. If csv file exists results are appended otherwise a new csv file is
    created.

    Parameters
    ----------
    df_results: PandasDataFrame
        A dataframe containing the results from model evaluation.
    """
    path_to_file = os.path.join('..', 'models', 'classification_results.csv')
    if os.path.exists(path_to_file):
        with open(path_to_file, 'a') as f:
            df_results.to_csv(f, header=False)
    else:
        df_results.to_csv(path_to_file)

