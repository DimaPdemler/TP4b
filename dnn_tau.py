from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Dropout
from typing import List
from tensorflow import function

class Dnn_tau(Model):
	def __init__(self, depths=[10,10,10,10], activation='relu', output_size = 1):
		super().__init__()

		
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
      
	@function
	def call(self, x):
		print(type(x))
		for layer in self.layers_:
			x = layer(x)
		return self.output_(x)

