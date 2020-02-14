# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 19:15:58 2020

@author: Jesus
"""
import numpy as np
from scipy.stats import mode
from sklearn.linear_model import LogisticRegression

class BalancedEnsemble:
    """
    Class that will fit a LogisticRegression model to several splits of the dataset.
    Each split is made of a different subset (non-overlapping) of the majority 
    class instances plus all minority class instances, such that both are roughly
    the same size at each split.
    
    Implements the fit() and predict() methods "à la" scikit-learn so that it can
    be used as a model from that library.
    """
    def __init__(self):
        self.model = []
    def fit(self, X, y):
        n = len(y)//sum(y)
        X_maj, X_min = X[y == 0], X[y == 1]
        _, y_min = y[y == 0], y[y == 1]
        for xm in np.array_split(X_maj,n):
            X_this = np.concatenate((xm,X_min))
            y_this = np.concatenate((np.zeros(xm.shape[0]),y_min))
            self.model.append(LogisticRegression().fit(X_this,y_this))
        return self
    def predict(self, X):
        preds = []
        for mdl in self.model:
            preds.append(mdl.predict(X))
        return mode(preds)[0][0] #Most common label