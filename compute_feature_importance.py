from utils import split_dataset
from metrics_features import fi_perm
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

learning_features = ['charge_1', 'charge_2', 'charge_3', 'pt_1',
       'pt_2', 'pt_3', 'pt_MET', 'eta_1', 'eta_2', 'eta_3', 'mass_1', 'mass_2',
       'mass_3', 'deltaphi_12', 'deltaphi_13', 'deltaphi_23', 'deltaphi_1MET',
       'deltaphi_2MET', 'deltaphi_3MET', 'deltaphi_1(23)', 'deltaphi_2(13)',
       'deltaphi_3(12)', 'deltaphi_MET(12)', 'deltaphi_MET(13)',
       'deltaphi_MET(23)', 'deltaphi_1(2MET)', 'deltaphi_1(3MET)',
       'deltaphi_2(1MET)', 'deltaphi_2(3MET)', 'deltaphi_3(1MET)',
       'deltaphi_3(2MET)', 'deltaeta_12', 'deltaeta_13', 'deltaeta_23',
       'deltaeta_1(23)', 'deltaeta_2(13)', 'deltaeta_3(12)', 'deltaR_12',
       'deltaR_13', 'deltaR_23', 'deltaR_1(23)', 'deltaR_2(13)',
       'deltaR_3(12)', 'pt_123', 'mt_12', 'mt_13', 'mt_23', 'mt_1MET',
       'mt_2MET', 'mt_3MET', 'mt_1(23)', 'mt_2(13)', 'mt_3(12)', 'mt_MET(12)',
       'mt_MET(13)', 'mt_MET(23)', 'mt_1(2MET)', 'mt_1(3MET)', 'mt_2(1MET)',
       'mt_2(3MET)', 'mt_3(1MET)', 'mt_3(2MET)', 'mass_12', 'mass_13',
       'mass_23', 'mass_123', 'Mt_tot', 'HNL_CM_angle_with_MET_1',
       'HNL_CM_angle_with_MET_2', 'W_CM_angle_to_plane_1',
       'W_CM_angle_to_plane_2', 'W_CM_angle_to_plane_with_MET_1',
       'W_CM_angle_to_plane_with_MET_2', 'HNL_CM_mass_1', 'HNL_CM_mass_2',
       'HNL_CM_mass_with_MET_1', 'HNL_CM_mass_with_MET_2', 'W_CM_angle_12',
       'W_CM_angle_13', 'W_CM_angle_23', 'W_CM_angle_1MET', 'W_CM_angle_2MET',
       'W_CM_angle_3MET', 'mass_hyp']
selection = learning_features + ['signal_label']
selection_all = selection + ['channel']

tf.keras.models.load_model(complete_path('saved_models/TEST9_global_v4_all_channels_depth_10'))

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

# # for channel in channel_indices:
# for channel in ['tee']:
#     train_channel = train[train['channel'] == channel_indices[channel]]
#     print("Debug : ", len(train_channel['event']))
#     val_channel = val[val['channel'] == channel_indices[channel]]
#     test_channel = test[test['channel'] == channel_indices[channel]]
#     meas_channel = meas[meas['channel'] == channel_indices[channel]]
#     print(len(train_channel['event']) + len(val_channel['event']) + len(test_channel['event']) + len(meas_channel['event']))

#     compute_fi_perm(train_channel, val_channel, test_channel, selection, "TEST9_global_v4", channel)

# compute_fi_perm(train, val, test, selection_all, "TEST9_global_v4", 'all_channels')

