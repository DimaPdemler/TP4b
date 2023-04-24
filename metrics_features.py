import tensorflow as tf
import pandas as pd
import numpy as np

def fi_perm(model, test_df, input_vars, keys_shuffle):

    if type(keys_shuffle) != list:
        keys_shuffle = [keys_shuffle]

    x_test = test_df[input_vars]
    y_test = x_test.pop('signal_label').astype(float)
    
    loss_no_shuffle, _ = model.evaluate(x_test, y_test, sample_weight=test_df['genWeight'], verbose=0)
    for key in keys_shuffle:
        x_test[key] = np.random.permutation(x_test[key])
    loss_shuffle, _ = model.evaluate(x_test, y_test, sample_weight=test_df['genWeight'], verbose=0)

    return loss_no_shuffle, loss_shuffle