# Imports
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from typing import List
from typing import TypeVar
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
PandasSeries = TypeVar('pandas.core.series.Series')
SklearnClassifier = TypeVar('sklearn.svm._classes.SVC')

# ROC Function
def plot_roc_curve(
        X_test: PandasDataFrame,
        y_test: PandasSeries,
        clfs: List[SklearnClassifier],
        clfs_names: List[str]):
    """
    Function that creates the Receiver-operator-characteristic curve (ROC-Curve) for all specified models.

    Parameters
    ----------
    X_test: PandasDataFrame
        Dataframe that contains all feature space of test data
    y_test: PandasSeries
        Series that contains the ground-truth labels of the test data
    clfs: List[SklearnClassifier]
        A list containing all models that should be plotted on the ROC-Curve.
    clfs_names: List[str]
        A list containing all the model names that should be shown in legend of the plot.

    """

    clf_probas = []
    for clf in clfs:
        y_pred_proba = clf.predict_proba(X_test)
        clf_probas.append(y_pred_proba)

    random_probs = [0 for _ in range(len(y_test))]

    # calculate ROC Curve
    # For the Random Model
    plt.figure(figsize=(10, 8))
    random_fpr, random_tpr, _ = roc_curve(y_test, random_probs)
    plt.plot(random_fpr, random_tpr, linestyle='--', label='Random')

    # For the models
    colors = ['b', 'g', 'o', 'r']
    for i in range(clf_probas):
        model_fpr, model_tpr, _ = roc_curve(y_test, clf_probas[i])
        plt.plot(model_fpr, model_tpr, marker='.', label=clfs_names[i], color=colors[i])
    # Plot the roc curve for the model and the random model line
    # Create labels for the axis
    plt.title('ROC-Curve', fontsize=22)
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.xticks(fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.yticks(fontsize=14)

    # show the legend
    plt.legend()
    # show the plot
    plt.show()
