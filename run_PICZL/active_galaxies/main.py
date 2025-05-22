

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
#catalog_data_url = '/home/wroster/learning-photoz/PICZL_new/gather_images/data_euc/EUC_CPT_LS10_all_cols.fits'
#catalog_data_url = '/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/FLASH_30/FLASH_30_PICZL_ready.fits'
catalog_data_url = '/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/FLASH_comp/combined_FLASH_PICZL_ready.fits'

#image_data_url = "/home/wroster/learning-photoz/PICZL_new/gather_images/data_euc/"
image_data_url = "/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/FLASH_comp/"


with tf.device('/GPU:0'):

	dataset, image_data = fetch_all_inputs(catalog_data_url, image_data_url, False, 2000)

	#Preprocess catalog
	dataset = run_all_preprocessing(dataset)
	combined_non_2D_features, index, labels = grab_features(dataset)


	#Preparing images for ML model
	images, images_col = stack_images(image_data)


	# ---------------------------------------------------------------
	# Import and run models
	# ---------------------------------------------------------------



	# Load the model with the custom loss function
	model_path = '/home/wroster/learning-photoz/PICZL_new/output/set_2/'
	models = ['nll/models/0_1/model_G=3_B=256_lr=0.001.h5',
	          'nll/models/0_1/model_G=4_B=512_lr=0.001.h5',
	          'nll/models/0_1/model_G=5_B=512_lr=0.001.h5',
	          'nll/models/0_1/model_G=5_B=512_lr=0.00115.h5',
	          'nll/models/0_2/model_G=7_B=256_lr=0.00085.h5',
	          'crps/models/0_1/model_G=3_B=512_lr=0.00115.h5',
	          'crps/models/0_2/model_G=10_B=512_lr=0.00085.h5',
	          'crps/models/0_3/model_G=11_B=512_lr=0.001.h5',
	          'crps/models/1_1/model_G=17_B=256_lr=0.00115.h5',
	          'crps/models/1_2/model_G=19_B=256_lr=0.001.h5']

	weights = [0.7903637084444504, 0.18859199953280673, 0.2392967739669823,0.7994359423449, 0.5882119658624398, 0.8726770934471257,0.1700934005996512,0.2588162827956706,0.8733373254080712, 0.044601371708622585]
	normalized_weights = weights/np.sum(weights)

	all_pdfs = []

	for x in range(0, len(models)):

		model = load_model(model_path+models[x] , compile=False)
		preds = model.predict([images, images_col, combined_non_2D_features])
		pdf_scores, samples = get_pdfs(preds, len(dataset), 4001)
		all_pdfs.append(pdf_scores)


	norm_ens_pdfs, ens_modes, lower_1sig, upper_1sig, lower_3sig, upper_3sig, area_in_interval = ensemble_pdfs(normalized_weights, all_pdfs, samples)

	print(ens_modes)
	#np.savez(image_data_url + 'pdf_data_act.npz', samples=samples, pdf_scores=norm_ens_pdfs)
	#sys.exit()

	#sys.exit()

	#outlier_frac, accuracy = calculate_metrics(ens_modes, labels)
	#print('outlier frac: '+str(outlier_frac))
	#print('accuracy: ' +str(accuracy))


	# ---------------------------------------------------------------
	# Save results
	# ---------------------------------------------------------------
	error_results = batch_classify(samples[0], norm_ens_pdfs)
	# Extract best_interval bounds
	l1s = [res['best_interval'][0] for res in error_results]  # Lower bounds
	u1s = [res['best_interval'][1] for res in error_results]  # Upper bounds
	degeneracy = [res['degeneracy'] for res in error_results]
	# Optional: get z_peak as "ens_modes" and anything else you need
	#ens_modes = [res['z_peak'] for res in error_results]


	pwd = image_data_url
	catalog_name = 'FLASH_'
	append_output(dataset, pwd, catalog_name, ens_modes, l1s, u1s, area_in_interval, degeneracy)














