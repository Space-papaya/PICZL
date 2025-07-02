

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



def arrange_tt_features(images, images_col, combined_non_2D_features, index, labels):

	'''
	Input: Takes all training variables (numerical, 2D SCF and images (including colour images) as well as labels and an index
	Output: Returns

	This function takes all relevant features for training and splits them in test and train sets. It returns all necessary features for later use in training the model.
	'''
	#random shuffle
	random_state=42

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


