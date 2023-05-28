import tensorflow as tf
import pandas as pd
from numpy.random import permutation
from copy import deepcopy
from sklearn.feature_selection import mutual_info_regression
from numpy import linspace, array, sqrt, sum
from tqdm import tqdm

def fi_perm(model, test_df, input_vars, keys_shuffle):
    """
    Arguments : 
        -model : trained DNN
        -test_df : Test dataset containing the learning features, the weights and the signal labels
        -input_vars : list of feature names + name of signal label column
        -keys_shuffle : list of features to shuffle
    Output : 
        Loss for the evaluation of the model on the test dataset with the columns in keys_shuffle randomly permuted
    """

    if type(keys_shuffle) != list:
        keys_shuffle = [keys_shuffle]

    test = deepcopy(test_df)
    x_test = test[input_vars]
    y_test = x_test.pop('signal_label').astype(float)
    
    for key in keys_shuffle:
        test[key] = permutation(test[key])
        x_test = test[input_vars]
        y_test = x_test.pop('signal_label').astype(float)
    results = model.evaluate(x_test, y_test, sample_weight=test_df['genWeight'], verbose=0)

    return results[0]


def fi_mutual_info(model, test_df, input_vars):
    """
    Arguments : 
        -model : trained DNN
        -test_df : Test dataset containing the learning features, the weights and the signal labels
        -input_vars : list of feature names + name of signal label column
    Output : 
        Mutual information between each feature and the model prediction

    Warning : This function isn't tested
    """
    x_test = test_df[input_vars]
    x_test_arr = x_test.values
    y_test = x_test.pop('signal_label').astype(float)
    y_pred = model.predict(x_test)

    mis = mutual_info_regression(x_test_arr, y_pred)
    return mis

def poisson_significance(scores, signal_label, weights, bins):
    """
    Arguments : 
        -scores : values of the variable to study
        -signal_label : signal label (1 for signal, 0 for background) corresponding to the events described by the scores
        -weights : weights of the individual scores
        -bins : 
            If int : number of descriminating thresholds which will be linearly aranged between min(scores) and max(scores)
            If np.array or list : discriminating thresholds
    Output : 
        -significances : s/sqrt(b) for each discriminating threshold, except the ones for wihch there's no event in the
                         signal/background category, or the ones for which the error on the background is greater 
                         than 10%    
        -x0s : descriminating thresholds at which the significances where computed
    """
    if type(bins) == int:
        bins = linspace(min(scores), max(scores), bins)

    scores_s = scores[signal_label==1]
    scores_b = scores[signal_label==0]
    weights_s = weights[signal_label==1]
    weights_b = weights[signal_label==0]

    significances = []
    x0s = []
    
    for x0 in tqdm(bins):

        weights_s_x0 = weights_s[scores_s > x0]
        weights_b_x0 = weights_b[scores_b > x0]
        scores_s_x0 = scores_s[scores_s > x0]
        scores_b_x0 = scores_b[scores_b > x0]

        S = sum(weights_s_x0)
        B = sum(weights_b_x0)
        
        if S<=0 or B<=0:
            continue

        err_B = sqrt(sum(weights_b_x0**2))/B
        if err_B > 0.1:
            continue
        
        significances.append(S/sqrt(B))
        x0s.append(x0)

    return array(significances), array(x0s)