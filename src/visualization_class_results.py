# Imports
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from typing import List, Tuple
from utils import load_model


# ROC Function
def plot_roc_curve(y_test: List[int], y_pred_probas: List[List[float]], clfs_names: List[Tuple[str, str]]):
    """
    Function that creates the Receiver-operator-characteristic curve (ROC-Curve) for all specified models.

    Parameters
    ----------
    y_test: List[int]
        Ground-truth labels for each sample.
    y_pred_probas: List[List[float]]
        A list containing np matrices with prediction probabilities for each classifier.
    clfs_names: List[Tuple[str, str]]
        A list containing all model names and if pca was used or not that should be shown in legend of the plot.

    """

    random_probs = [0 for _ in range(len(y_test))]

    # calculate ROC Curve
    # For the Random Model
    plt.figure(figsize=(10, 8))
    random_fpr, random_tpr, _ = roc_curve(y_test, random_probs)
    plt.plot(random_fpr, random_tpr, linestyle='--', label='Random')

    # For the models
    colors = ['b', 'g', 'k', 'r']
    for i in range(len(y_pred_probas)):
        model_fpr, model_tpr, _ = roc_curve(y_test, y_pred_probas[i][:, 1])
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

    # save ROC curve
    plt.savefig(os.path.join('..', 'models', 'roc_curve.png'))
    # show the plot
    plt.show()
