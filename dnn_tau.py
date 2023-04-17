from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Dropout, Embedding
from typing import List
from tensorflow import function, concat

class Dnn_tau(Model):
	def __init__(self, input_vars, depths=[10,10,10,10], n_channels=5, ch_emb_dim=3, activation='relu', output_size = 1):
		"""
		Arguments : 
			-input_vars : list of keys of the dataset in the right order
			-depths : len(depths) = number of hidden layers. The ith element of depths corresponds to the depth of the ith hidden layer
			-n_channels : number of possible channel considered for the events in the datasets
			-ch_emb_dim : dimension of the embedding layer for the various channels
			-actiavtion : activation fonction between the hidden layers
			-output_size : dimension of the output
		"""
		super().__init__()

		self.input_vars = input_vars
		
		if type(depths) != list:
			depths = [depths]
		layers_ = list()

		for depth in depths:
			layers_.append(Dense(units=depth, activation=activation))
			layers_.append(Dropout(0.2))
		self.layers_ = layers_
		if output_size == 1:
			last_act = 'sigmoid'
		else:
			last_act = 'softmax'
		self.output_ = Dense(units=output_size, activation=last_act)
		self.emb = Embedding(n_channels, ch_emb_dim)
      
	@function
	def call(self, x):
		# Embedding of the channel column
		ch_var_idx = self.input_vars.index('channel')
		x = concat([x[:,:ch_var_idx], self.emb(x[:,ch_var_idx]), x[:,ch_var_idx+1:]], axis=1)

		# Apply the layers to the data tensor
		for layer in self.layers_:
			x = layer(x)
		return self.output_(x)

