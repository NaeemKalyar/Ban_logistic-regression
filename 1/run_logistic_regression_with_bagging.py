"""
main code that you will run
"""
import pandas as pd
import numpy as np
from linear_model import LogisticRegression
from ensemble import BaggingClassifier
from data_handler import load_dataset, split_dataset
from metrics import precision_score, recall_score, f1_score, accuracy

if __name__ == '__main__':
    # data load
    X, y = load_dataset('data_banknote_authentication.csv')

    # split train and test
    X_train, y_train, X_test, y_test = split_dataset(X, y)

    # training
    #params = dict()
    base_estimator = LogisticRegression()
    classifier = BaggingClassifier(base_estimator, 9,X_train,X_test,y_test,y_train)   
    classifier.fit(X_train, y_train)

    # testing
    y_pred = classifier.predict(X_test)

    # performance on test set
    print('Accuracy ', accuracy(y_true=y_test, y_pred=y_pred))
    print('Recall score ', recall_score(y_true=y_test, y_pred=y_pred))
    print('Precision score ', precision_score(y_true=y_test, y_pred=y_pred))
    print('F1 score ', f1_score(y_true=y_test, y_pred=y_pred))
