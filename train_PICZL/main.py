#PICZL for mixed galaxies
###################

# ---------------------------------------------------------------
# 1. Import data preprocessing functions
# ---------------------------------------------------------------

import tensorflow as tf
from astropy.table import Table
import pickle
import numpy as np
import sys
import os

sys.path.append('/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/inactive_galaxies/')

from utilities import gpu_configuration
from utilities.load_data import *
from utilities.handling_images import *
from preprocessing_catalog.clean_and_extend import *
from preprocessing_catalog.feature_downselection import *
from model.get_model import *
from model.train_test_data import *
from model.train_model_0_1 import *
from utilities.loss_functions import *
from post_processing.distributions import *
#from post_processing.output import *


#Load input data
data_url = "/home/wroster/learning-photoz/PICZL_OZ/train_PICZL/data"
catalog_url = data_url + '/TS.fits'
image_url = data_url + '/combined/'


with tf.device('/GPU:0'):

	dataset, image_data = fetch_all_inputs(catalog_url, image_url, True, 20000)

	#Preprocess catalog
	dataset = run_all_preprocessing(dataset)
	labels = np.array(dataset['Z'])
	catalog_features, index  = grab_features(dataset)

	#Preparing images for ML model
	images, images_col = stack_images(image_data)

	print(images.shape)
	print(images_col.shape)
	print(catalog_features.shape)

	#Prepares relevant settings for the CNN training
	print('>> Preparing train and test data ...')
	train_images, test_images, train_labels, test_labels, train_features, test_features, train_ind, test_ind, train_col_images, test_col_images \
	= arrange_tt_features(images, images_col, catalog_features, index, labels)


	# ---------------------------------------------------------------
	# Import and run models
	# ---------------------------------------------------------------


	# Training Hyperparameters
	loss_func = NLL_loss #NLL_loss  #crps_loss
	epochs = 400

	batch_sizes = [512]
	num_gaussian = [8]
	learning_rates = [0.0005]


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
				save_dir = "/home/wroster/learning-photoz/PICZL_OZ/train_PICZL/output/test/models/0_1"

				# Create and train multiple models
				model = compile_model(catalog_features.shape[1], num_gauss, learning_rate, loss_func)
				history, model = train_model(model, epochs, batch_size, learning_rate, loss_func,
							train_images, train_col_images, train_features, train_labels, test_images, test_col_images, test_features, test_labels)

				print(f"Model {model_counter} trained. Validation Loss: {min(history.history['val_loss'])}")

				# Save the model to a file
				model_file = os.path.join(save_dir, f"model_G={num_gauss}_B={batch_size}_lr={learning_rate}.h5")
				model.save(model_file)
				preds = model.predict([test_images, test_col_images, test_features])
				all_predictions.append(preds)

				# Save the training history and configurations
				config = {'gmm_components': num_gauss, 'batch_size': batch_size, 'learning_rate': learning_rate}
				history_and_config = {'config': config, 'history': history.history}
				all_histories_and_configs.append(history_and_config)





	# Save all_histories_and_configs using pickle
	with open('/home/wroster/learning-photoz/PICZL_OZ/train_PICZL/output/test/hists/0_1/hist.pkl', 'wb') as pickle_file:
		pickle.dump(all_histories_and_configs, pickle_file)


	# Save the all_preds list to a file using pickle
	with open('/home/wroster/learning-photoz/PICZL_OZ/train_PICZL/output/test/preds/0_1/preds.pkl', 'wb') as file:
		pickle.dump(all_predictions, file)




#############################################################################

with open('/home/wroster/learning-photoz/PICZL_OZ/train_PICZL/output/test/preds/0_1/preds.pkl', 'rb') as pickle_file:
	loaded_preds = pickle.load(pickle_file)
	loaded_preds = np.array(loaded_preds)


for i in tqdm(range(len(loaded_preds))):

	pdf_scores, samples = get_pdfs(loaded_preds[i], len(test_labels), 4001)
	modes = get_point_estimates(pdf_scores, samples)
	outlier_frac, accuracy = calculate_metrics(modes, test_labels)

	print('outlier frac: '+str(outlier_frac))
	print('accuracy: ' +str(accuracy))


################################
################################



