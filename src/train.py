import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from random import choices
!pip install datatable > /dev/null
import datatable as dt
from sklearn import impute
from models import create_mlp
from feature_engineering import clean_data
import gc

SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

if __name__ == "__main__":
	# Comment these out if ran once
	df = pd.read_csv("../input/train.csv")
	df = clean_data(df)
	df.to_csv('../output/cleaned_df.csv', index=False)
	# Then uncomment this
	#df = pd.read_csv("../output/cleaned_df.csv")
	# define params
	batch_size = 4096
	hidden_units = [150, 150, 150]
	dropout_rates = [0.20, 0.20, 0.20, 0.20]
	label_smoothing = 1e-2
	learning_rate = 3e-3

	# build model
	clf = create_mlp(
		X_train.shape[1], 5, hidden_units, dropout_rates, label_smoothing, learning_rate
		)
	# fit model
	clf.fit(X_train, y_train, epochs=100, batch_size=batch_size)

	models = []
	models.append(clf)

	# evaluate
	test_pred = clf.predict(X_test)
	test_pred = np.rint(test_pred)
	test_acc = np.sum(test_pred == y_test)/(y_test.shape[0]*5)
	print("test accuracy: " + str(test_acc))

	# Random Search CV 
	### IMPORTANT: Memory intensive. Will likely give back out of memory error.
	batch_size = 5000
	hidden_units = [(150, 150, 150), (100,100,100), (200,200,200)]
	dropout_rates = [(0.25, 0.25, 0.25, 0.25), (0.3,0.3,0.3,0.3)]
	epochs = 100
	num_columns = len(features)
	num_labels = 5

	mlp_CV = KerasClassifier(build_fn=create_mlp, epochs=epochs, batch_size=batch_size, verbose=1)

	param_distributions = {'hidden_units':hidden_units, 'learning_rate':[1e-3, 1e-4], 
					  'label_smoothing':[1e-2, 1e-1], 'dropout_rates':dropout_rates,
						  'num_columns': [len(features)], 'num_labels': [5]}

	random_cv = RandomizedSearchCV(estimator=mlp_CV, 
								   param_distributions=param_distributions, n_iter=5,
								   n_jobs=-1, cv=3, random_state=42)

	random_cv.fit(X_train, y_train, callbacks=[EarlyStopping(patience=10)])
	best_model = random_cv.best_estimator_.model
	models.append(best_model)

	# save models to models folder.
	i = 0
	for model in models:
		path = "../models/model_" + str(i)
		model.save(path)