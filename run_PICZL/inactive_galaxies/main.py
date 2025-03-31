

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

with tf.device('/GPU:0'):

	#Load input data
	catalog_data_url_og = '/home/wroster/learning-photoz/PICZL_galaxies/samples/TS_40489.fits'
	catalog_data_url = '/home/wroster/learning-photoz/PICZL_galaxies/samples/COSMOS_NGOETZ_310.fits'
	image_data_url = "/home/wroster/learning-photoz/PICZL_new/gather_images/data_cosmos_og/"


	dataset, image_data = fetch_all_inputs(catalog_data_url, image_data_url, False, 100, catalog_data_url_og)
	# Unpack dictionary keys into local variables
	locals().update(image_data)


	#Preprocess catalog
	original_dataset, dataset = run_all_preprocessing(dataset)

