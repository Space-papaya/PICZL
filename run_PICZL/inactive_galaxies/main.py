

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
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from utilities import gpu_configuration
from utilities.load_data import *
#from piczl.utilities import *
from utilities.handling_images import *
from preprocessing_catalog.clean_and_extend import *
from preprocessing_catalog.feature_downselection import *
from tensorflow.keras.models import load_model
from utilities.loss_functions import *
from post_processing.distributions import *
from post_processing.output import *

#Load input data
#catalog_data_url = '/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/COSMOS_NGOETZ_216.fits'
#catalog_data_url = '/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/FLASH_30/FLASH_30_PICZL_ready.fits'
catalog_data_url = '/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/FLASH_comp/combined_FLASH_PICZL_ready.fits'

#image_data_url = "/home/wroster/learning-photoz/PICZL_new/gather_images/data_cosmos_og/"
image_data_url = "/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/FLASH_comp/"


with tf.device('/GPU:0'):

	dataset, image_data = fetch_all_inputs(catalog_data_url, image_data_url, False, 20)
	#dataset, image_data = load_data.fetch_all_inputs(catalog_data_url, image_data_url, True, 20)

	#Preprocess catalog
	dataset = run_all_preprocessing(dataset)
	combined_non_2D_features, index = grab_features(dataset)

	#Preparing images for ML model
	images, images_col = stack_images(image_data)

	#print(images.shape)
	#print(images_col.shape)
	#print(combined_non_2D_features.shape)


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
		pdf_scores, samples = distributions.get_pdfs(preds, len(dataset), 4001)
		all_pdfs.append(pdf_scores)


	norm_ens_pdfs, ens_modes, lower_1sig, upper_1sig, lower_3sig, upper_3sig, area_in_interval = distributions.ensemble_pdfs(normalized_weights, all_pdfs, samples)

	#Saving pdfs
	#np.savez(image_data_url + 'pdf_data_inact.npz', samples=samples, pdf_scores=norm_ens_pdfs)
	#sys.exit()

	# Sample statistics if spec-z available
	#outlier_frac, accuracy = calculate_metrics(ens_modes, labels)
	#print('outlier frac: '+str(outlier_frac))
	#print('accuracy: ' +str(accuracy))


	# ---------------------------------------------------------------
	# Save results
	# ---------------------------------------------------------------
	error_results = distributions.batch_classify(samples[0], norm_ens_pdfs)
	# Extract best_interval bounds
	l1s = [res['best_interval'][0] for res in error_results]  # Lower bounds
	u1s = [res['best_interval'][1] for res in error_results]  # Upper bounds
	degeneracy = [res['degeneracy'] for res in error_results]
	# Optional: get z_peak as "ens_modes" and anything else you need
	#ens_modes = [res['z_peak'] for res in error_results]

	sys.exit()
	pwd = image_data_url
	catalog_name = 'FLASH_'

	output.append_output(dataset, pwd, catalog_name, ens_modes, l1s, u1s, area_in_interval, degeneracy)


