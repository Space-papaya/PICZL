import tensorflow as tf
import pickle
import numpy as np
import sys
from tensorflow.keras import backend as K
import gc
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from piczl.utilities import *



def run_models(loss_func, epochs, batch_sizes, num_gaussian, learning_rates, version, features, \
		train_images, train_col_images, train_features, train_labels, test_images, test_col_images, test_features, test_labels):

	# Training Hyperparameters
	if loss_func == loss_functions.crps_loss:
		lf = 'CRPS'
	else:
		lf = 'NLL'

	# Initialize a list to store all training histories and configurations
	all_histories_and_configs = []
	all_predictions = []
	all_train_predictions = []

	model_counter=0
	# Loop over hyperparameter values
	for num_gauss in num_gaussian:
		for batch_size in batch_sizes:
			for learning_rate in learning_rates:

				model_counter = model_counter+1
				# Define a directory to save the models
				save_dir = "/home/wroster/learning-photoz/PICZL_OZ/models/from_train/" + lf

				# Before starting a new model, clear the previous session:
				tf.keras.backend.clear_session()

				# Create and train multiple models
				model = get_model.compile_model(features.shape[1], num_gauss, learning_rate, loss_func)
				history, model = train_model.train_model(model, epochs, batch_size, learning_rate, loss_func, version,
							train_images, train_col_images, train_features, train_labels, test_images, test_col_images, test_features, test_labels)

				print(f"Model {model_counter} trained. Validation Loss: {min(history.history['val_loss'])}")

				# Save the model to a file
				model_file = os.path.join(save_dir + "/models/" + version, f"model_G={num_gauss}_B={batch_size}_lr={learning_rate}.h5")
				model.save(model_file)
				preds = model.predict([test_images, test_col_images, test_features])
				all_predictions.append(preds)

				# Save the training history and configurations
				config = {'gmm_components': num_gauss, 'batch_size': batch_size, 'learning_rate': learning_rate}
				history_and_config = {'config': config, 'history': history.history}
				all_histories_and_configs.append(history_and_config)

				# After model is saved and predictions done:
				del model
				K.clear_session()
				gc.collect()


	return all_predictions, all_histories_and_configs


