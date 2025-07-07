import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
MODEL_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../models'))

from piczl.config.training_config import *
from piczl.utilities import *


def run_trainer(catalog_path, image_path, mode, sub_sample, max_sources):
	"""
	Run redshift estimation for a given catalog using a specified configuration.

	Parameters:
	catalog_path (str): Path to catalog file
	image_path (str): Path to image data
	max_sources (int): Limit number of sources (for testing)
	"""

	with tf.device('/GPU:0'):
		psf = False if mode == 'active' else True

		dataset, image_data = fetch_inputs.fetch_all_data(catalog_path, image_path, exec='train', psf=psf, sub_sample_yesno=sub_sample, sub_sample_size=max_sources)
		dataset = cat_preprocessing.run_all_preprocessing(dataset)
		print(dataset)
		features, index = feature_extraction.grab_features(dataset, mode)
		images, images_col = image_processing.stack_images(image_data)
		labels = dataset['z']


		train_images, test_images, train_labels, test_labels, train_features, test_features, train_ind, test_ind, train_col_images, test_col_images \
		= train_test_data.arrange_tt_features(images, images_col, features, index, labels)


		print(type(train_images), train_images.dtype, train_images.shape)
		print(type(train_col_images), train_col_images.dtype, train_col_images.shape)
		print(type(train_features), train_features.dtype, train_features.shape)
		print(type(train_labels), train_labels.dtype, train_labels.shape)


		run_models(
		loss_func=loss_functions.crps_loss,
		epochs=50,
		batch_sizes=[8],
		num_gaussian=[5],
		learning_rates=[0.001],
		version='0_1',
		features=features,
		train_images=train_images,
		train_col_images=train_col_images,
		train_features=train_features,
		train_labels=train_labels,
		test_images=test_images,
		test_col_images=test_col_images,
		test_features=test_features,
		test_labels=test_labels
		)



