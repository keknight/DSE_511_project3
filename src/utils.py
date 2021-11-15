import os
import pickle
import numpy as np


def data_loader(pca: bool=False) -> np.ndarray:
    """
    Function to load the pickled data.

    Parameters
    ----------
    pca: bool
        variable determining which type of processed data is requests, e.g., 'scaled' or 'scaled_pca'

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

