# -*- coding: utf-8 -*-
# @Author: foxwy
# @Date:   2019-10-18 16:39:15
# @Last Modified by:   WY
# @Last Modified time: 2022-03-07 21:27:53

import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


def dnn_regressor(indata, outdata, layer_nums, epochs, batch_size):
	insize = len(indata[0])
	outsize = 1 # len(outdata[0])

	# define base model
	def baseline_model():
		# create model
		model = Sequential()
		model.add(Dense(layer_nums[0], input_dim=insize, kernel_initializer='normal', activation='relu'))
		for layer_num in layer_nums[1:]:
			model.add(Dense(layer_num, kernel_initializer='normal', activation='relu'))
		model.add(Dense(outsize, kernel_initializer='normal'))

		# Compile model
		model.compile(loss='mean_squared_error', optimizer='adam')
		return model

	# fix random seed for reproducibility
	seed = 7
	np.random.seed(seed)

	# evaluate model with standardized dataset
	estimators = []
	estimators.append(('standardize', StandardScaler()))
	estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=epochs, batch_size=batch_size, verbose=0)))
	pipeline = Pipeline(estimators)
	#kfold = KFold(n_splits=5, random_state=seed)
	results = cross_val_predict(pipeline, indata, np.array(outdata))
	mse = mean_squared_error(results, np.array(outdata))
	print('mse:', mse)
	'''scores = cross_val_score(pipeline, np.array(indata), np.array(outdata), cv=kfold, scoring='neg_mean_squared_error')
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))'''