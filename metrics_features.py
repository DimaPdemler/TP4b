import tensorflow as tf
import pandas as pd
from numpy.random import permutation
from copy import deepcopy
from sklearn.feature_selection import mutual_info_regression

def fi_perm(model, test_df, input_vars, keys_shuffle):

    if type(keys_shuffle) != list:
        keys_shuffle = [keys_shuffle]

    test = deepcopy(test_df)
    x_test = test[input_vars]
    y_test = x_test.pop('signal_label').astype(float)
    
    for key in keys_shuffle:
        test[key] = permutation(test[key])
        x_test = test[input_vars]
        y_test = x_test.pop('signal_label').astype(float)
    loss_shuffle, _ = model.evaluate(x_test, y_test, sample_weight=test_df['genWeight'], verbose=0)

    return loss_shuffle


def fi_mutual_info(model, test_df, input_vars):
    x_test = test_df[input_vars]
    x_test_arr = x_test.values
    y_test = x_test.pop('signal_label').astype(float)
    y_pred = model.predict(x_test)

    mis = mutual_info_regression(x_test_arr, y_pred)
    return mis

        