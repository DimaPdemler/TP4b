import pickle
import pandas as pd
import tensorflow as tf
import os
from dnn_tau import Dnn_tau
from train_model import train_model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
depths = [1,2,3,4,5,6,7,9,11,13,15]

name = 'sel2'
features_file = 'saved_results/features_and_label_sel2'

current_directory = os.path.dirname(os.path.abspath(__file__))

def complete_path(rel_path):
    path = os.path.join(current_directory, rel_path)
    return path

train = pd.read_pickle(complete_path('extracted_data/TEST9_global_v4_og_and_norm_weights_train'))
val = pd.read_pickle(complete_path('extracted_data/TEST9_global_v4_og_and_norm_weights_val'))

while True:
    with open(complete_path(features_file), 'rb') as file:
        features = pickle.load(file)

    for depth in depths:
        print(depth)
        train_model(depth, train, val, features, 'saved_models/average_models/', f'TEST9_{name}_depth_{depth}')