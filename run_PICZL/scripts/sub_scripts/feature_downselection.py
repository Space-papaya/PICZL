

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


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------



def grab_features(dataset):
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
	features = features.drop(['FULLID','RA','DEC', "Cat", 'type','TS_ID'], axis=1)

	l=0
	for i in features.columns:
		l=l+1
		print(l,i)

	sys.exit()

	#Splitting features
	features_dchisq = np.array(features.iloc[:, 0:5])
	features_snr = np.array(features.iloc[:,[5,7,9,11,13,15,17,19]])
	features_dered_flux = np.array(features.iloc[:,[6,8,10,12]]) #only WISE, after adding model image for g,r,i,z
	features_frac_flux = np.array(features.iloc[:, 37:45])
	features_psf_size = np.array(features.iloc[:, 49:53])
	features_shape_e1 = np.array(features.iloc[:, 61])
	features_shape_e1_ivar = np.array(features.iloc[:, 62])
	features_shape_e2 = np.array(features.iloc[:, 63])
	features_shape_e2_ivar = np.array(features.iloc[:, 64])
	features_type = np.array(features.iloc[:, 65:70])
	features_col = np.array(features.iloc[:, 76:92]) #removed griz and w1-w4 colours, substituted by images


	#Normalize all non spatially connected features
	feature_arrays = ['features_dchisq', 'features_snr', 'features_dered_flux', 'features_frac_flux', 'features_psf_size',\
			 'features_shape_e1', 'features_shape_e1_ivar', 'features_shape_e2', 'features_shape_e2_ivar']


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

