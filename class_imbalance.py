# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:06:44 2020

@author: Jesus
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import make_pipeline

from BalancedEnsemble import BalancedEnsemble
from aux_functions import check_class_imbalance, report_results

if __name__ == "__main__":
    ###########################################################################
    
    # Data
    X, y = make_classification(n_classes=2, weights=[0.9,0.1], 
                               n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        stratify=y)
    
    # Models
    mdl1 = LogisticRegression()
    mdl2 = BalancedEnsemble()
    
    # Under/Over-Sampling 
    us = RandomUnderSampler()
    os = RandomOverSampler()
    tl = TomekLinks()
    smt = SMOTE()
    st = SMOTETomek()
    
    # Pipelines
    baseline = make_pipeline(mdl1)
    rus = make_pipeline(us, mdl1)
    ros = make_pipeline(os, mdl1)
    tomek = make_pipeline(tl, mdl1)
    smote = make_pipeline(smt, mdl1)
    smtk = make_pipeline(st, mdl1)
    be = make_pipeline(mdl2)
    
    ###########################################################################
    
    # Check imbalance in datasets
    print(f"Original imbalance: {check_class_imbalance(y):.2%}")
    print(f"Train Set imbalance: {check_class_imbalance(y_train):.2%}")
    print(f"Test Set imbalance: {check_class_imbalance(y_test):.2%}")
    
    # Run each pipeline and evaluate it
    report_results(baseline, X_train, X_test, y_train, y_test, 'Baseline')
    report_results(rus, X_train, X_test, y_train, y_test, 'Random Under Sampling')
    report_results(ros, X_train, X_test, y_train, y_test, 'Random Over Sampling')
    report_results(smote, X_train, X_test, y_train, y_test, 'SMOTE')
    report_results(smote, X_train, X_test, y_train, y_test, 'SMOTE')
    report_results(smtk, X_train, X_test, y_train, y_test, 'SMOTE+Tomek Links' )
    report_results(be, X_train, X_test, y_train, y_test, 'Balanced Ensemble')
    