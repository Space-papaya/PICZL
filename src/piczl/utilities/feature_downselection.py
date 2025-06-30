

'''
#####  FEATURE SELECTION #####

This file holds functions to preprare the input features from the catalogue for the CNN.

The catalogue on which these functions can be run requires following characteristics:

        - Needs to follow the pre-processing performed in script "clean_and_extend"
        - ...

'''


# Importing libraries and dependancies
import numpy as np
from numpy import load
import pandas as pd
import sys

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------



def grab_features(dataset, mode):
	'''
	Input: dataset, feature names for aperture colours LS10 and WISE
	Output: Numerical, non-co-dependant scalar features, aperture arrays of photometry, ivariance and residuals for LS10 and WISE
	as well as aperture colour arrays for LS10 and WISE

	This function removes undesired features which can't or should not be used for training. It returns subsets of spatialy connected and independant features.
	'''

	#Make copy of dataset to work on features
	features = dataset.copy()
	index = features.index
	#labels = np.array(features['Z'])

	#Remove features
	features = features.drop(['FULLID','Z','RA','DEC', "Cat", 'type','TS_ID'], axis=1)

	# Set which features to select
	if mode != 'active':

		features_dchisq = np.array(features.iloc[:, 0:5])
		features_snr = np.array(features.iloc[:,[5,17,29,41,53,71,89,107]])
		features_dered_flux = np.array(features.iloc[:,[6,18,30,42]])
		features_frac_flux = np.array(features.iloc[:, 133:141])
		features_psf_size = np.array(features.iloc[:, 141:145])
		features_shape_e1 = np.array(features.iloc[:, 145])
		features_shape_e1_ivar = np.array(features.iloc[:, 146])
		features_shape_e2 = np.array(features.iloc[:, 147])
		features_shape_e2_ivar = np.array(features.iloc[:, 148])
		features_type = np.array(features.iloc[:, 150:155])
		features_col = np.array(features.iloc[:, 213:229])

		#Normalize all non spatially connected features
		feature_arrays = ['features_dchisq', 'features_snr', 'features_dered_flux', 'features_frac_flux', 'features_psf_size',\
				 'features_shape_e1', 'features_shape_e1_ivar', 'features_shape_e2', 'features_shape_e2_ivar']


	else:

		#Splitting features
		features_dchisq = np.array(features.iloc[:, 0:5])
		features_snr = np.array(features.iloc[:,[5,17,29,41,53,71,89,107]])
		features_dered_flux = np.array(features.iloc[:,[6,18,30,42]])
		features_frac_flux = np.array(features.iloc[:, 133:141])
		features_type = np.array(features.iloc[:, 150:155])
		features_col = np.array(features.iloc[:, 213:235])


		#Normalize all non spatially connected features
		feature_arrays = ['features_dchisq', 'features_snr', 'features_dered_flux', 'features_frac_flux']


	scaled_features = {}

	# Loop through the feature arrays, scale, and normalize them
	for feature in feature_arrays:
		feature_name = feature.replace("features_", "")
		feature_data = np.array(eval(feature))
		global_mean = np.mean(feature_data)
		global_std = np.std(feature_data)

		scaled = (feature_data - global_mean) / global_std
		normalized = (scaled - np.min(scaled)) / (np.max(scaled) - np.min(scaled))

		# Reshape normalized features if they are 1D
		if normalized.ndim == 1:
			normalized = normalized.reshape(-1, 1)

		# Store in dictionary with a relevant key name
		scaled_features[f'scaled_feature_{feature_name}'] = normalized


	#Returning an array featuring all normalized features with no spatial relation
	combined_non_2D_features = np.concatenate((list(scaled_features.values()) + [features_col, features_type]), axis=1)
	print('>> Feature extraction completed')

	return combined_non_2D_features, index

