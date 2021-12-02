# Imports
#import pandas as pd
import matplotlib.pyplot as plt
import os

from typing import TypeVar
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
PandasSeries = TypeVar('pandas.core.series.Series')

from utils import load_model

from sklearn.inspection import permutation_importance

def feature_importance(
        X_test: PandasDataFrame,
        y_test: PandasSeries,
        clf_v: str,
        pca: bool=False) -> PandasDataFrame:
    """
    Function that looks at feature importance.

    Parameters
    ----------
    X_test: PandasDataFrame
        Dataframe that contains all features of test data
    y_test: PandasSeries
        Series that contains the ground-truth labels of the test data
    clf_v: str
        Variable that indicates the type of initialized classifier ('nb', 'svc', 'rf', 'dt')
    pca: bool
        Boolean variable indicating if pca data used or not.

    Returns
    -------
    Graphs that show the feature importance of each model plus whether the data was PCA or Scaled
    """

    # load the model
    clf = load_model(clf_v=clf_v, pca=pca)

    if pca is True:
    	d = 'PCA'
    else:
    	d = 'Scaled'

    r = permutation_importance(clf, X_test, y_test, n_repeats=30, random_state=0, n_jobs=-1)

    sorted_idx = r.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(r.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx])
    ax.set_title("Permutation Importance: " + str(clf_v) + ' ' + str(d))
    fig.tight_layout()
    plt.savefig(os.path.join('..', 'reports', 'figures', 'permutation_figure_' + str(clf_v) + '_' + str(d) + '.png'))
    plt.show()
