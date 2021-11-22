"""
Title: Binary Classification - Naive Bayes and Support Vector Machine
Author: Christoph Metzner

This file contains source code to perform a binary classification using the machine learning algorithms naive bayes
and support vector machine. The classification is performed on normalized data and data with reduced dimensions using
principal component analysis.

"""
import pandas as pd

import os
from utils import load_model

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
        time: float,
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
    time: float
        Average computing time of each model during hyperparameter tuning with cross-validation gridsearch
    clf_v: str
        Variable that indicates the type of initialized classifier ('nb', 'svc')
    pca: bool
        Boolean variable indicating if pca data used or not.

    Returns
    -------
    PandasDataFrame
        A dataframe containing the performance metric results
    """
    # load the model
    clf = load_model(clf_v=clf_v, pca=pca)


    y_pred = clf.predict(X_test)  # get prediction for test dataset
    y_pred_proba = clf.predict_proba(X_test)

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

    if pca:
        pca_s = 'pca_true'
    else:
        pca_s = 'pca_false'
    results = [[clf_v, pca_s, f1, precision, recall, tp, fp, fn, tn, time]]
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
                 'tn',
                 'time'])

    save_results(df_results=df_results, file_name='classification_results.csv')

    return df_results, y_pred_proba


def save_results(df_results: PandasDataFrame, file_name: str):
    """
    Function that saves the results to a csv file. If csv file exists results are appended otherwise a new csv file is
    created.

    Parameters
    ----------
    df_results: PandasDataFrame
        A dataframe containing the results from model evaluation. The dataframe should contain values in the
        following order: 'model', 'pca', 'f1_score', 'precision', 'recall', 'tp', 'fp', 'fn', 'tn'.
    file_name: str
        Name of file
    """
    path_to_file = os.path.join('..', 'models', file_name)
    clf_v = df_results.loc[0, 'model']
    pca = df_results.loc[0, 'pca']

    if os.path.exists(path_to_file):
        df_res_exist = pd.read_csv(path_to_file)
        # Check if results for model and data-preprocessing method already exist in csv file
        if df_res_exist.loc[(df_res_exist['model'] == clf_v) & (df_res_exist['pca'] == pca)].any().all():
            print('Model evaluation results already stored in file')
        else:
            with open(path_to_file, 'a') as f:
                df_results.to_csv(f, header=False)
    else:
        df_results.to_csv(path_to_file)

