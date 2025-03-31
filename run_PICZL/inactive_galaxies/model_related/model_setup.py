

'''
#####  MANAGE_CNN_setup #####

This file holds function to preprare the input featuress for CNN training.

The variables on which these functions can be run require following characteristics:

        - all images .npy files need to be of same dimensions
        - all variables need to have the same array length
        - ...
'''



# Importing libraries and dependancies
import numpy as np
from numpy import load
import pandas as pd
from sklearn.model_selection import train_test_split



# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------



def arrange_tt_features(images_g, images_r, images_i, images_z, images_gr, images_gi, images_gz, images_ri, images_rz, images_iz, \
        mod_g, mod_r, mod_i, mod_z, mod_gr, mod_gi, mod_gz, mod_ri, mod_rz, mod_iz, ap_ims_g_band_res, ap_ims_r_band_res, ap_ims_i_band_res, ap_ims_z_band_res, \
        ap_ims_g_band_ivar, ap_ims_r_band_ivar, ap_ims_i_band_ivar, ap_ims_z_band_ivar, ap_ims_w1_band_res, ap_ims_w2_band_res, ap_ims_w3_band_res, ap_ims_w4_band_res, \
        ap_ims_w1_band_ivar, ap_ims_w2_band_ivar, ap_ims_w3_band_ivar, ap_ims_w4_band_ivar, ap_ims_g_band, ap_ims_r_band, ap_ims_i_band, ap_ims_z_band, ap_ims_w1_band, \
        ap_ims_w2_band, ap_ims_w3_band, ap_ims_w4_band, ap_ims_gi_band, ap_ims_gr_band, ap_ims_gz_band, ap_ims_ri_band, ap_ims_rz_band, ap_ims_iz_band, ap_ims_w12_band, \
        ap_ims_w13_band, ap_ims_w14_band, ap_ims_w23_band, ap_ims_w24_band, ap_ims_w34_band, psf_g, psf_r, psf_i, psf_z, combined_non_2D_features, index, labels):

	'''
	Input: Takes all training variables (numerical, 2D SCF and images (including colour images) as well as labels and an index
	Output: Returns

	This function takes all relevant features for training and splits them in test and train sets. It returns all necessary features for later use in training the model.
	'''
	#Colour image stack selection
	images_col = np.stack((images_gr, images_gi, images_gz, images_ri, images_rz, images_iz,\
			ap_ims_gr_band,ap_ims_gi_band,ap_ims_gz_band,ap_ims_ri_band,ap_ims_rz_band,ap_ims_iz_band,\
			ap_ims_w12_band,ap_ims_w13_band,ap_ims_w14_band,ap_ims_w23_band,ap_ims_w24_band,ap_ims_w34_band,\
			mod_gr, mod_gi,mod_gz,mod_ri,mod_rz, mod_iz), axis=-1)

	#Flux image stack selection
	images = np.stack((images_g, images_r, images_i, images_z,\
			ap_ims_g_band,ap_ims_r_band,ap_ims_i_band,ap_ims_z_band,\
			ap_ims_w1_band,ap_ims_w2_band,ap_ims_w3_band,ap_ims_w4_band,\
			ap_ims_g_band_res, ap_ims_r_band_res, ap_ims_i_band_res, ap_ims_z_band_res,\
			ap_ims_w1_band_res,ap_ims_w2_band_res,ap_ims_w3_band_res,ap_ims_w4_band_res,\
			ap_ims_g_band_ivar,ap_ims_r_band_ivar,ap_ims_i_band_ivar,ap_ims_z_band_ivar,\
			ap_ims_w1_band_ivar, ap_ims_w2_band_ivar, ap_ims_w3_band_ivar, ap_ims_w4_band_ivar,\
			mod_g, mod_r, mod_i, mod_z, psf_g, psf_r, psf_i, psf_z), axis=-1)

	###
	#testing

	#images = np.stack((ap_ims_g_band,ap_ims_r_band,ap_ims_i_band,ap_ims_z_band,\
        #               ap_ims_w1_band,ap_ims_w2_band,ap_ims_w3_band,ap_ims_w4_band), axis=-1)

	###

	#split the images, labels and features into training and test sets
	random_state=42 #number of shuffles, 42 works well as default

	#Split flux images, labels, catalog features, indices
	train_images, test_images, train_labels, test_labels, train_features, test_features, train_ind, test_ind = train_test_split(
	 images, labels, combined_non_2D_features, index, test_size=0.2, random_state=random_state)

	#Split image colours data
	train_col_images, test_col_images = train_test_split(images_col, test_size=0.2, random_state=random_state)


	print("Train flux images shape: "+str(train_images.shape))
	print("Train colour images shape: "+str(train_col_images.shape))
	print("Train catalog features shape: "+str(train_features.shape))
	print("Train labels shape: "+str(train_labels.shape))


	return train_images, test_images, train_labels, test_labels, train_features, test_features, train_ind, test_ind, train_col_images, test_col_images

        # -------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------


