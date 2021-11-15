import os
import pickle
import numpy as np
from typing import TypeVar
SklearnClassifier = TypeVar('sklearn.svm._classes.SVC')


def data_loader(pca: bool=False) -> np.ndarray:
    """
    Function to load the pickled data.

    Parameters
    ----------
    pca: bool
        variable determining if pca was used or not.

    Returns
    -------
    df_train: PandasDataFrame
        A dataframe containing the scaled or scaled + pca reduced training data and ground-truth labels
    df_test: PandasDataFrame
        A dataframe containing the scaled or scaled + pca reduced testing data and ground-truth labels
    """
    if pca:
        with open(os.path.join('..', 'data', 'processed', 'train_scaled_pca.pkl'), 'rb') as input_file:
            train = pickle.load(input_file)
        with open(os.path.join('..', 'data', 'processed', 'test_scaled_pca.pkl'), 'rb') as input_file:
            test = pickle.load(input_file)

    elif pca is False:
        with open(os.path.join('..', 'data', 'processed', 'train_scaled.pkl'), 'rb') as input_file:
            train = pickle.load(input_file)
        with open(os.path.join('..', 'data', 'processed', 'test_scaled.pkl'), 'rb') as input_file:
            test = pickle.load(input_file)

    X_train = train[0]
    y_train = train[1]

    X_test = test[0]
    y_test = test[1]

    return X_train, y_train, X_test, y_test


def save_model(clf_best: SklearnClassifier, clf_v: str, pca: bool):
    """
    Function that pickles estimator.
    Parameters
    ----------
    clf_best: SklearnClassifier
        Best estimator for a given algorithm and hyper-parameter space
    clf_v: str
        Variable that indicates the type of initialized classifier ('nb', 'svc')
    pca: bool
        Variable indicating whether scaled or scaled + pca data is used
    """
    if pca:
        pca_s = '_pca'
    else:
        pca_s = ''
    with open(os.path.join('..', 'models', f'trained_{clf_v}{pca_s}_model.pkl'), 'wb') as f:
        pickle.dump(clf_best, f)


def load_model(clf_v: str, pca: bool):
    """
    Function that loads a pickled estimators.
    Parameters
    ----------
    clf_v: str
        Variable that indicates the type of initialized classifier ('nb', 'svc').
    pca: bool
        Variable indicating whether scaled or scaled + pca data is used.

    Returns
    -------
    SklearnClassifier
        Loads pickled estimator for a given algorithm and data preprocessing method.
    """
    if pca:
        pca_s = '_pca'
    else:
        pca_s = ''
    # loading the best model from hyper-parameter tuning and training
    with open(os.path.join('..', 'models', f'trained_{clf_v}{pca_s}_model.pkl'), 'rb') as f:
        clf = pickle.load(f)
    return clf
