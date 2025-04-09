

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
from utilities.handling_images import *
from preprocessing_catalog.clean_and_extend import *
from preprocessing_catalog.feature_downselection import *
from tensorflow.keras.models import load_model
from utilities.loss_functions import *
from post_processing.distributions import *
from post_processing.output import *

#Load input data
#catalog_data_url = '/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/COSMOS_NGOETZ_216.fits'
catalog_data_url = '/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/FLASH_30/FLASH_30_PICZL_ready.fits'

#image_data_url = "/home/wroster/learning-photoz/PICZL_new/gather_images/data_cosmos_og/"
image_data_url = "/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/FLASH_30/"


with tf.device('/GPU:0'):

	dataset, image_data = fetch_all_inputs(catalog_data_url, image_data_url, False, 20)


	#Preprocess catalog
	dataset = run_all_preprocessing(dataset)
	combined_non_2D_features, index = grab_features(dataset)
	print(combined_non_2D_features.shape)

	#Preparing images for ML model
	images, images_col = stack_images(image_data)

	print(images[0])
	print(images_col[0])
	print(combined_non_2D_features[0])

	# ---------------------------------------------------------------
	# Import and run models
	# ---------------------------------------------------------------


	# Load the model with the custom loss function
	model_path = '/home/wroster/learning-photoz/PICZL_galaxies/output/psf/'
	models = ['crps/models/0_3/model_G=11_B=512_lr=0.0002.h5',
		'crps/models/0_1/model_G=4_B=256_lr=0.0002.h5',
		'crps/models/0_1/model_G=5_B=256_lr=0.00035.h5',
		'crps/models/0_2/model_G=7_B=256_lr=0.0005.h5',
		'nll/models/0_1/model_G=5_B=512_lr=0.0005.h5',
		'nll/models/0_1/model_G=4_B=512_lr=0.0005.h5',
		'nll/models/0_1/model_G=5_B=256_lr=0.0005.h5',
		'nll/models/0_1/model_G=3_B=512_lr=0.00035.h5'
		]

	weights= [0.08227451309829409, 0.17245762536202783, 0.02718044442985964, 0.11249663791786871, 0.2592408249176275, 0.1408404956224673, 0.02917528832932573, 0.1763341703225292]
	normalized_weights = weights/np.sum(weights)

	all_pdfs = []

	for x in range(0, len(models)):

		model = load_model(model_path+models[x] , compile=False)
		preds = model.predict([images, images_col, combined_non_2D_features])
		pdf_scores, samples = get_pdfs(preds, len(dataset), 4001)
		all_pdfs.append(pdf_scores)


	norm_ens_pdfs, ens_modes, lower_1sig, upper_1sig, lower_3sig, upper_3sig = ensemble_pdfs(normalized_weights, all_pdfs, samples)
	print(ens_modes)


	#outlier_frac, accuracy = calculate_metrics(ens_modes, labels)
	#print('outlier frac: '+str(outlier_frac))
	#print('accuracy: ' +str(accuracy))


	# ---------------------------------------------------------------
	# Save results
	# ---------------------------------------------------------------


	pwd = image_data_url
	catalog_name = 'FLASH_30'

	append_output(dataset, pwd, catalog_name, ens_modes, lower_1sig, upper_1sig, lower_3sig, upper_3sig)


