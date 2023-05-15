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

current_directory = os.path.dirname(os.path.abspath(__file__))

def complete_path(rel_path):
    path = os.path.join(current_directory, rel_path)
    return path

data = pd.read_pickle(complete_path('extracted_data/TEST9_global_v4_all_normalized'))
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
selection = learning_features+['signal_label']

print(selection)

depths = [5,7,10,15]

def compute_fi_perm(data, name, channel):
    print(len(data['event']))

    train, val, test, meas = split_dataset(data)

    x_train = train[selection]
    x_test = test[selection]
    x_val = val[selection]
    x_meas = meas[selection]

    label_train = x_train.pop('signal_label').astype(float)
    label_val = x_val.pop('signal_label').astype(float)
    label_test = x_test.pop('signal_label').astype(float)
    label_meas = x_meas.pop('signal_label').astype(float)
    x_train = x_train.astype(float)
    x_test = x_test.astype(float)
    x_val= x_val.astype(float)
    x_meas = x_meas.astype(float)

    losses = []
    for depth in depths:
        widths = [len(learning_features)*2]*depth
        model = Dnn_tau(learning_features, widths=widths)
        model.compile(loss='binary_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath="./saved_models/checkpoint",
            monitor = "val_loss",
            save_best_only = True
        )
        history = model.fit(x_train, label_train, sample_weight=train['genWeight'], validation_data=(x_val, label_val), epochs=100000, verbose=1,  # type: ignore
                            batch_size = 300, callbacks=[early_stopping, checkpoint])
        model = tf.keras.models.load_model('./saved_models/checkpoint')
        model.save(complete_path('saved_models/'+ name + '_' + channel + '_depth_{}'.format(depth)))
        # Save history
        filename = complete_path('saved_history/'+ name + '_' + channel + '_depth_{}'.format(depth))
        with open(filename, "wb") as file:
            pickle.dump(history.history, file) # type: ignore

        loss, _ = model.evaluate(x_val, label_val)
        losses.append(loss)
    
    losses = np.array(losses)
    idx_best = np.argmin(losses)
    model = tf.keras.models.load_model(complete_path('saved_models/'+ name + '_' + channel + '_depth_{}'.format(depths[idx_best])))

    delta_loss_perm = []
    loss_no_shuffle = fi_perm(model, test, selection, [])
    for key in learning_features:
        loss_shuffle = fi_perm(model, test, selection, key)
        delta_loss_perm.append([loss_shuffle-loss_no_shuffle])
    delta_loss_perm = dict(zip(learning_features, delta_loss_perm))

    filename = complete_path("saved_results/" + name + "_loss_shuffle_" + channel)
    with open(filename, "wb") as file:
        pickle.dump(delta_loss_perm, file)

# for channel in channel_indices:
#     data_channel = data.loc[data['channel'] == channel_indices[channel]]
#     print(len(data_channel['event']))

#     compute_fi_perm(data_channel, "TEST9_global_v4", channel)

compute_fi_perm(data, "TEST9_global_v4", 'all_channels')

    