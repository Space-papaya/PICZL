

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

with tf.device('/GPU:0'):

	#Load input data
	catalog_data_url_og = '/home/wroster/learning-photoz/PICZL_galaxies/samples/TS_40489.fits'
	catalog_data_url = '/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/COSMOS_NGOETZ_216.fits'
	image_data_url = "/home/wroster/learning-photoz/PICZL_new/gather_images/data_cosmos_og/"


	dataset, image_data = fetch_all_inputs(catalog_data_url, image_data_url, False, 100, catalog_data_url_og)
	locals().update(image_data)
	# Create a list of the local variables
	variables = [globals()[var] for var in image_data.keys()]
	print(dataset)
	sys.exit()

	#Preprocess catalog
	original_dataset, dataset = run_all_preprocessing(dataset)
	#Extracts features from catalogue which are relevant for training
	print('>> Extracting relevant features ...')
	combined_non_2D_features, index, labels = grab_features(dataset)
	print('>> Feature extraction completed')

	train_images, test_images, train_labels, test_labels, train_features, test_features, train_ind, test_ind, train_col_images, test_col_images \
        = arrange_tt_features(*variables, combined_non_2D_features, index, labels)


	test_df = dataset.iloc[test_ind]
	print(test_df)
	#sys.exit()

	# ---------------------------------------------------------------
	# Import and run models
	# ---------------------------------------------------------------
