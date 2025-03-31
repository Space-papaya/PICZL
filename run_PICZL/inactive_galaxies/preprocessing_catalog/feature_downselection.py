

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
	labels = np.array(features['Z'])


	#Remove undesired features
	#non-trainable, amount: 11
	features = features.drop(['FULLID','Z','RA','DEC', "Cat", 'BRICKID','BRICKNAME','RELEASE','OBJID','type','TS_ID'], axis=1)
	#not helpful for redshift estimates based on scientific relevance, amount: 82
	features = features.drop(['RA_IVAR','DEC_IVAR','RCHISQ_G','RCHISQ_R', 'RCHISQ_I', 'RCHISQ_Z','RCHISQ_W1','RCHISQ_W2','RCHISQ_W3','RCHISQ_W4', \
				  'ANYMASK_G','ANYMASK_R','ANYMASK_I','ANYMASK_Z','ALLMASK_G','ALLMASK_R','ALLMASK_I','ALLMASK_Z','WISEMASK_W1','WISEMASK_W2',\
	                          'GALDEPTH_G','GALDEPTH_R','GALDEPTH_I','GALDEPTH_Z','WISE_COADD_ID','FIBERFLUX_G','FIBERFLUX_R','FIBERFLUX_I','FIBERFLUX_Z',\
				  'FIBERTOTFLUX_G','FIBERTOTFLUX_R','FIBERTOTFLUX_I','FIBERTOTFLUX_Z', 'REF_CAT','REF_ID','REF_EPOCH','GAIA_PHOT_G_MEAN_MAG',\
				  'GAIA_PHOT_G_MEAN_FLUX_OVER_ERROR','GAIA_PHOT_BP_MEAN_MAG','GAIA_PHOT_BP_MEAN_FLUX_OVER_ERROR','GAIA_PHOT_RP_MEAN_MAG',\
				  'GAIA_PHOT_RP_MEAN_FLUX_OVER_ERROR','GAIA_ASTROMETRIC_EXCESS_NOISE', 'GAIA_DUPLICATED_SOURCE','GAIA_PHOT_BP_RP_EXCESS_FACTOR',\
				  'GAIA_ASTROMETRIC_SIGMA5D_MAX','GAIA_ASTROMETRIC_PARAMS_SOLVED','PARALLAX','PARALLAX_IVAR','PMRA','PMRA_IVAR','PMDEC','PMDEC_IVAR',\
				  'MASKBITS','FITBITS','SERSIC','SERSIC_IVAR','ORIG_TYPE','EBV','FRACMASKED_G','FRACMASKED_R','FRACMASKED_I', 'FRACMASKED_Z'],axis=1)



	#Splitting features
	features_dchisq = np.array(features.iloc[:, 0:5])
	features_snr = np.array(features.iloc[:,[5,7,9,11,13,15,17,19]])
	features_dered_flux = np.array(features.iloc[:,[6,8,10,12]]) #removed g,r,i,z, after adding model image
	features_nobs = np.array(features.iloc[:, 29:37])
	features_frac_flux = np.array(features.iloc[:, 37:45])
	features_fracin = np.array(features.iloc[:, 45:49])
	features_psf_size = np.array(features.iloc[:, 49:53])
	features_psf_depth = np.array(features.iloc[:, 53:59])
	features_shape_r = np.array(features.iloc[:, 59])
	features_shape_r_ivar = np.array(features.iloc[:, 60])
	features_shape_e1 = np.array(features.iloc[:, 61])
	features_shape_e1_ivar = np.array(features.iloc[:, 62])
	features_shape_e2 = np.array(features.iloc[:, 63])
	features_shape_e2_ivar = np.array(features.iloc[:, 64])
	features_type = np.array(features.iloc[:, 65:70])
	features_col = np.array(features.iloc[:, 76:92]) #remove griz and w1-w4 colours
	#features_maskbits = np.array(features.iloc[:, 98:])


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


	return combined_non_2D_features, index, labels

