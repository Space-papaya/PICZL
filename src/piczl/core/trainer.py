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
		# Set whether to include PSF images based on mode
		psf = False if mode == 'active' else True

		dataset, image_data = load_data.fetch_all_inputs(catalog_path, image_path, psf=psf, sub_sample_yesno=sub_sample, sub_sample_size=max_sources)
		dataset = clean_and_extend.run_all_preprocessing(dataset)
		features, index = feature_downselection.grab_features(dataset, mode)
		images, images_col = handling_images.stack_images(image_data)
		labels = dataset['Z']


		train_images, test_images, train_labels, test_labels, train_features, test_features, train_ind, test_ind, train_col_images, test_col_images \
		= train_test_data.arrange_tt_features(images, images_col, features, index, labels)


		predictions, histories = run_models(
		loss_func=loss_functions.crps_loss,
		epochs=50,
		batch_sizes=[512, 1024],
		num_gaussian=[5, 6, 7],
		learning_rates=[0.001, 0.002],
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




