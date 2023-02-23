import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import os


def error_type_row(trained: pd.DataFrame, my_label: str = "computed_label", answer_label: str = "Label") -> tuple[pd.DataFrame, pd.DataFrame]:

    def helper(mine, theirs):
        if theirs == mine:
            return "Correct Answer"
        elif bool(theirs) and not bool(mine):
            return "False Negative"
        else:
            return "False Positive"
    trained["error_type"] = trained.apply(lambda x: helper(x[my_label], x[answer_label]), axis=1)

    return trained, trained.loc[(trained["error_type"] == "False Positive") | (trained["error_type"] == "False Negative")] #df[df["error_type"].isin(("False Negative", "False Positive"))]


def make_confusion_matrix(trained: pd.DataFrame) -> tuple[tuple[int], tuple[int]]:
    os.chdir("/Users/NoahRipstein/PycharmProjects/Bayes email 2/visualizations")
    y_true = trained["Label"].values
    y_pred = trained["computed_label"].values.astype(int)
    return plot_confusion_matrix(y_true, y_pred, classes=["Spam", "Ham"], save=True)


def plot_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, save=False) -> tuple[tuple[int], tuple[int]]:
    """Makes a labelled confusion matrix comparing predictions and ground truth labels.

    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.

    Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).

    Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

    Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
    """
    plt.style.use("default")
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize it
    n_classes = cm.shape[0]  # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)  # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # If there are labels for the classes, add them
    if classes is not None:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),  # create enough axis slots for each class
           yticks=np.arange(n_classes),
           xticklabels=labels,  # axes will be labeled with class names (if they exist) or ints
           yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=text_size)

    if save:
        plt.savefig("confusion matrix.png", dpi=300)
    plt.show()
    return cm[0], cm[1]
