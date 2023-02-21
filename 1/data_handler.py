#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
import numpy as np

def load_dataset(input_file_name):
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    # todo: implement
    data = pd.read_csv(input_file_name)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y


def split_dataset(X, y, test_size=1-0.2, shuffle=True):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.
    X_train = X.sample(frac=test_size, random_state=25)
    X_test = X.drop(X_train.index)
    y_train = y[X_train.index]
    y_test = y.drop(X_train.index)
    X_train, y_train, X_test, y_test = X_train, y_train, X_test, y_test
    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implement
    X_sample=X.sample(frac=1,random_state=30)
    y_sample=y[X_sample.index]
    X_sample, y_sample = X_sample, y_sample
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample
   

