# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 19:17:05 2020

@author: Jesus
"""

import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, roc_auc_score

def check_class_imbalance(labels):
    """
    Returns % of class labeled as '1' in binary classification.
    """
    return sum(labels)/len(labels)

def plot_confusion_matrix(cm, title='Confusion Matrix', target_names=['0','1']):
    """
    Plot confusion matrix in a nicer way.
    
    Adapted from https://stackoverflow.com/a/50386871
    """

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)


    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.show()
    
def report_results(pipeline, X_train, X_test, y_train, y_test, name='Confusion Matrix'):
    """
    Reports confusion matrix and Cohen Kappa score of the pipeline trained
    on "train" dataset and tested on "test" dataset.
    """
    y_hat = pipeline.fit(X_train, y_train).predict(X_test)
    plot_confusion_matrix(confusion_matrix(y_test, y_hat), name)
    print(f"{name} - ROC AUC Score: {roc_auc_score(y_test, y_hat):.2}") #1 is best