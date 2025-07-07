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

	if loss_func == loss_functions.crps_loss:
		lf = 'CRPS'
	else:
		lf = 'NLL'

	all_histories_and_configs = []
	all_predictions = []
	all_train_predictions = []

	model_counter=0
	for num_gauss in num_gaussian:
		for batch_size in batch_sizes:
			for learning_rate in learning_rates:

				model_counter = model_counter+1
				# Before starting a new model, clear the previous session:
				tf.keras.backend.clear_session()

				# Create and train multiple models
				model = get_model.compile_model(features.shape[1], num_gauss, learning_rate, loss_func)
				history, model, checkpoint_dir = train_model.train_model(model, epochs, batch_size, learning_rate, loss_func, version,
							train_images, train_col_images, train_features, train_labels, test_images, test_col_images, test_features, test_labels)

				print(f"Model {model_counter} trained. Validation Loss: {min(history.history['val_loss'])}")

				# Make predictions
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



	# Save all_histories_and_configs using pickle
	with open(checkpoint_dir + '/hist.pkl', 'wb') as pickle_file:
		pickle.dump(all_histories_and_configs, pickle_file)


	# Save the all_preds list to a file using pickle
	with open(checkpoint_dir + '/preds.pkl', 'wb') as file:
		pickle.dump(all_predictions, file)


