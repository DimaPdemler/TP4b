from dnn_tau import Dnn_tau
import tensorflow as tf
from pickle import dump
from copy import deepcopy
from os import listdir
from fnmatch import filter
from utils import isolate_int

def train_model(depth, train, val, vars, save_path, model_name): #, save_path_history):
    weight_name = 'weightNorm'

    if 'signal_label' not in vars:
        vars.append('signal_label')

    n = len(vars)
    x_train = train[vars]
    x_val = val[vars]
    label_train = x_train.pop('signal_label').astype(float)
    label_val = x_val.pop('signal_label').astype(float)
    features = deepcopy(vars)
    features.remove('signal_label')

    widths = [n*2]*depth
    model = Dnn_tau(features, widths=widths)
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy'],
                  weighted_metrics=['accuracy']
                  )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=7)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="./saved_models/checkpoint",
        monitor = "val_loss",
        save_best_only = True
    )

    print('model defined')

    history = model.fit(x_train, label_train, sample_weight=train[weight_name], validation_data=(x_val, label_val, val[weight_name]), epochs=100000, verbose=1,  # type: ignore
                        batch_size = 350, callbacks=[early_stopping, checkpoint])
    
    model = tf.keras.models.load_model('./saved_models/checkpoint')

    print(save_path)
    print(model_name)
    existing_models = filter(listdir(save_path), model_name+'*')
    print(existing_models)
    if len(existing_models) == 0:
        suffix = '_n_0'
    else:
        ns = []
        for name in existing_models:
            n = isolate_int(name, '_')[1]
            ns.append(n)
        n = max(ns)
        suffix = f'_n_{n+1}'
    model.save(save_path+model_name+suffix)
    # # Save history
    # with open(save_path_history, "wb") as file:
    #     dump(history.history, file) # type: ignore