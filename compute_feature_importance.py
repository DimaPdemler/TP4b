from utils import split_dataset
from TP4b.metrics import fi_perm
import pandas as pd
import pickle
from data_extractor import output_vars_v4
from dnn_tau import Dnn_tau
import tensorflow as tf
import numpy as np
import os
import tensorflow as tf
from copy import deepcopy


current_directory = os.path.dirname(os.path.abspath(__file__))

def complete_path(rel_path):
    path = os.path.join(current_directory, rel_path)
    return path

train = pd.read_pickle(complete_path('extracted_data/TEST9_global_v4_train'))
val = pd.read_pickle(complete_path('extracted_data/TEST9_global_v4_val'))
test = pd.read_pickle(complete_path('extracted_data/TEST9_global_v4_test'))
meas = pd.read_pickle(complete_path('extracted_data/TEST9_global_v4_meas'))

file_path_idx = complete_path('extracted_data/TEST9_global_v4_all_normalized_channel_indices')
with open(file_path_idx, 'rb') as file:
    channel_indices = pickle.load(file)

with open(complete_path('saved_results/features&label_sel1'), 'rb') as file:
    features = pickle.load(file)

to_permute = deepcopy(features)
to_permute.remove('signal_label')
to_permute.remove('channel')

# model = tf.keras.models.load_model(complete_path('saved_models/TEST9_global_v4_all_channels_depth_10'))
model = tf.keras.models.load_model(complete_path('saved_models/TEST9_sel1_depth_2'))

def compute_fi_perm(model, test, var_names, to_permute, name, channel):

    features = deepcopy(var_names)
    features.remove('signal_label')

    delta_loss_perm = []
    loss_no_shuffle = fi_perm(model, test, var_names, [])
    for key in to_permute:
        loss_shuffle = fi_perm(model, test, var_names, key)
        delta_loss_perm.append([loss_shuffle-loss_no_shuffle])
    delta_loss_perm = dict(zip(features, delta_loss_perm))

    filename = complete_path("saved_results/" + name + "_loss_shuffle_" + channel)
    with open(filename, "wb") as file:
        pickle.dump(delta_loss_perm, file)

for channel in channel_indices:
    print(channel)
    test_channel = test[test['channel'] == channel_indices[channel]]
    compute_fi_perm(model, test_channel, features, to_permute, 'TEST9_sel1_test', channel)



