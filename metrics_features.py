import tensorflow as tf
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.feature_selection import mutual_info_regression

def fi_perm(model, test_df, input_vars, keys_shuffle):

    if type(keys_shuffle) != list:
        keys_shuffle = [keys_shuffle]

    x_test = test_df[input_vars]
    print("x_test defined")
    y_test = x_test.pop('signal_label').astype(float)
    print("y_test defined")
    
    loss_no_shuffle, _ = model.evaluate(x_test, y_test, sample_weight=test_df['genWeight'], verbose=0)
    print("Loss without shuffle evaluated")
    for key in keys_shuffle:
        x_test[key] = np.random.permutation(x_test[key])
    loss_shuffle, _ = model.evaluate(x_test, y_test, sample_weight=test_df['genWeight'], verbose=0)
    print("Loss with shuffle evaluated")

    return loss_no_shuffle, loss_shuffle
'''
To Do : 
    -adapt to have loss as a metric (search what 'evaluate' returns) and correct (what is called loss above is in fact the score I think)
    -Understand why the evaluation takes so long 
    -Check that the result in feature_importance_analysis.ipynb is correct
    and then work on fi_mutual_info
'''


def fi_mutual_info(model, test_df, input_vars):
    x_test = test_df[input_vars]
    x_test_arr = x_test.values
    y_test = x_test.pop('signal_label').astype(float)
    y_pred = model.predict(x_test)

    mis = mutual_info_regression(x_test_arr, y_pred)
    return mis

        