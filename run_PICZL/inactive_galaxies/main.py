

#PICZL for inactive galaxies
###################

# ---------------------------------------------------------------
# 1. Import data preprocessing functions
# ---------------------------------------------------------------

import tensorflow as tf
from astropy.table import Table
import pickle
import numpy as np
import sys

from utilities import gpu_configuration
from utilities.load_data import *
from preprocessing_catalog.clean_and_extend import *
from preprocessing_catalog.feature_downselection import *
from model_related.model_setup import *
from tensorflow.keras.models import load_model
from utilities.loss_functions import *
from post_processing.distributions import *


#Load input data
catalog_data_url = '/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/COSMOS_NGOETZ_216.fits'
image_data_url = "/home/wroster/learning-photoz/PICZL_new/gather_images/data_cosmos_og/"



with tf.device('/GPU:0'):

	dataset, image_data = fetch_all_inputs(catalog_data_url, image_data_url, False, 100)
	locals().update(image_data)
	# Create a list of the local variables
	variables = [globals()[var] for var in image_data.keys()]

	#Preprocess catalog
	dataset = run_all_preprocessing(dataset)
	#Extracts features from catalogue which are relevant for training
	print('>> Extracting relevant features ...')
	combined_non_2D_features, index, labels = grab_features(dataset)
	print('>> Feature extraction completed')


	# Separate variables based on whether they end with "col"
	col_variables = [var_name for var_name in image_data.keys() if var_name.endswith("col")]
	non_col_variables = [var_name for var_name in image_data.keys() if not var_name.endswith("col")]

	# Extract their values from globals()
	col_arrays = [globals()[var] for var in col_variables]
	non_col_arrays = [globals()[var] for var in non_col_variables]

	# Stack along the last axis
	images_col = np.stack(col_arrays, axis=-1)
	images = np.stack(non_col_arrays, axis=-1)

	# Print shapes to verify
	print(f"Stacked images_col shape: {images_col.shape}")
	print(f"Stacked images shape: {images.shape}")


	train_images, test_images, train_labels, test_labels, train_features, test_features, train_ind, test_ind, train_col_images, test_col_images \
	= arrange_tt_features(images, images_col, combined_non_2D_features, index, labels)


	# ---------------------------------------------------------------
	# Import and run models
	# ---------------------------------------------------------------


	# Load the model with the custom loss function
	model_path = '/home/wroster/learning-photoz/PICZL_galaxies/output/psf/'
	#model = load_model(model_path+'model_CRPS_G=10_B=512_lr=0.001_N:1.h5', custom_objects={'crps_loss': crps_loss})
	models = ['crps/models/0_3/model_G=11_B=512_lr=0.0002.h5',
		'crps/models/0_1/model_G=4_B=256_lr=0.0002.h5',
		'crps/models/0_1/model_G=5_B=256_lr=0.00035.h5',
		'crps/models/0_2/model_G=7_B=256_lr=0.0005.h5',
		'nll/models/0_1/model_G=5_B=512_lr=0.0005.h5',
		'nll/models/0_1/model_G=4_B=512_lr=0.0005.h5',
		'nll/models/0_1/model_G=5_B=256_lr=0.0005.h5',
		'nll/models/0_1/model_G=3_B=512_lr=0.00035.h5'
		]
#	weights = [1,1,1,1,1,1,1,1]
	weights= [0.08227451309829409, 0.17245762536202783, 0.02718044442985964, 0.11249663791786871, 0.2592408249176275, 0.1408404956224673, 0.02917528832932573, 0.1763341703225292]
	normalized_weights = weights/np.sum(weights)

	all_pdfs = []

	for x in range(0, len(models)):

		model = load_model(model_path+models[x] , compile=False)
		preds = model.predict([test_images, test_col_images, test_features])
		print(preds.shape)

		pdf_scores, samples = get_pdfs(preds, len(test_labels), 4001)
		all_pdfs.append(pdf_scores)



	norm_ens_pdfs, ens_modes, lower_1sig, upper_1sig, lower_3sig, upper_3sig = ensemble_pdfs(normalized_weights, all_pdfs, samples)


	#dataset['phz'] = ens_modes
	#print(dataset['phz'])
	#dataset['lower_1sig'] = lower_1sig
	#dataset['upper_1sig'] = upper_1sig
	#dataset['lower_3sig'] = lower_3sig
	#dataset['upper_3sig'] = upper_3sig

	#tab = Table.from_pandas(dataset)
	#tab.write(pwd + 'redshift_results.fits', overwrite=True)

	outlier_frac, accuracy = calculate_metrics(ens_modes, test_labels)
	print('outlier frac: '+str(outlier_frac))
	print('accuracy: ' +str(accuracy))











