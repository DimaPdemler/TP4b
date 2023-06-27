import pandas as pd
from copy import deepcopy
import tensorflow as tf
import numpy as np
import os
import pickle
from metrics import fi_perm

current_directory = os.path.dirname(os.path.abspath(__file__))
cdpath="/home/ddemler/dmitri_stuff/"


def complete_path(rel_path):
    path = os.path.join(current_directory, rel_path)
    return path

train = pd.read_pickle(cdpath + 'extracted_data/TEST10_v4_train')
val = pd.read_pickle(cdpath + 'extracted_data/TEST10_v4_val')
test = pd.read_pickle(cdpath + 'extracted_data/TEST10_v4_test')
meas = pd.read_pickle(cdpath + 'extracted_data/TEST10_v4_meas')

file_path_idx = cdpath + 'extracted_data/TEST10_channel_indices'
with open(file_path_idx, 'rb') as file:
    channel_indices = pickle.load(file)

selection = ['charge_1', 'charge_2', 'charge_3', 'pt_1',
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
       'W_CM_angle_3MET', 'mass_hyp', 'signal_label', 'channel']

features = deepcopy(selection)
features.remove('signal_label')
to_permute = deepcopy(features)
to_permute.remove('channel')
# to_permute.remove('mass_hyp')

depths = [5,7,10,15]

x_val = val[selection]
label_val = x_val.pop('signal_label').astype(float)

losses = []
for depth in depths:
    model = tf.keras.models.load_model('saved_models/TEST10_global_v4_all_channels_depth_{}'.format(depth))
    loss, _, _ = model.evaluate(x_val, label_val, sample_weight=val['genWeight'])
    losses.append(loss)
losses = np.array(losses)
idx_best = np.argmin(losses)
model = tf.keras.models.load_model('saved_models/TEST10_global_v4_all_channels_depth_{}'.format(depths[idx_best]))

selected_vars = []
# for channel in channel_indices:
for channel in ['ttm']:
    test_channel = test[test['channel'] == channel_indices[channel]]
    print(channel, " : ", len(test_channel['event']), " events")

    delta_loss_perm = []
    loss_no_shuffle = fi_perm(model, test, selection, [])
    for key in to_permute:
        print("\t", key, " permuting")
        loss_shuffle = fi_perm(model, test, selection, key)
        delta_loss_perm.append([loss_shuffle-loss_no_shuffle])
    delta_loss_perm = dict(zip(to_permute, delta_loss_perm))

    filename = complete_path("saved_results/TEST10_global_v4_loss_shuffle_all_channels_eval_" + channel)
    with open(filename, "wb") as file:
        pickle.dump(delta_loss_perm, file)
    
    sorted_fi = sorted(delta_loss_perm.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_fi]
    values = [item[1][0] for item in sorted_fi]

    threshold = values[0]/100
    sel_values = [v for v in values if v > threshold]
    sel_names = names[:len(sel_values)]

    selected_vars.extend(sel_names)

selected_vars = list(set(selected_vars))
filename = complete_path("saved_results/selected_vars_threshold_100")
with open(filename, "wb") as file:
    pickle.dump(selected_vars, file)

