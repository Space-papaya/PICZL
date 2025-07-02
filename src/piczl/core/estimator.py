import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
MODEL_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../models'))

from piczl.config.model_config import CONFIGS
from piczl.utilities import *


def run_estimation(catalog_path, image_path, mode, sub_sample, max_sources, pdf_samples=4001):
	"""
	Run redshift estimation for a given catalog using a specified configuration.

	Parameters:
	catalog_path (str): Path to catalog file
	image_path (str): Path to image data
	mode (str): One of 'active' or 'inactive'
	max_sources (int): Limit number of sources (for testing)
	pdf_samples (int): Number of PDF samples in redshift range [0,8]
	"""
	with tf.device('/GPU:0'):
		# Set whether to include PSF images based on mode
		psf = False if mode == 'active' else True

		dataset, image_data = load_data.fetch_all_inputs(catalog_path, image_path, psf=psf, sub_sample_yesno=sub_sample, sub_sample_size=max_sources)
		dataset = clean_and_extend.run_all_preprocessing(dataset)
		features, index = feature_downselection.grab_features(dataset, mode)
		images, images_col = handling_images.stack_images(image_data)

		config = CONFIGS[mode]
		model_files = config["model_files"]
		weights = np.array(config["model_weights"])
		normalized_weights = weights / np.sum(weights)

		all_pdfs = []
		for model_file in model_files:
			model_path = os.path.join(os.path.join(MODEL_BASE_DIR, mode), model_file)
			model = load_model(model_path, compile=False)
			preds = model.predict([images, images_col, features])
			pdfs, samples = distributions.get_pdfs(preds, len(dataset), pdf_samples)
			all_pdfs.append(pdfs)


		#Check details of what to output
		norm_ens_pdfs, z_modes, areas = distributions.ensemble_pdfs(normalized_weights, all_pdfs, samples)
		results = distributions.batch_classify(samples[0], norm_ens_pdfs)

		# Extract best_interval bounds
		l1s = [round(res['best_interval'][0],3) for res in results]
		u1s = [round(res['best_interval'][1],3) for res in results]
		degeneracy = [res['degeneracy'] for res in results]


		return z_modes, l1s, u1s, degeneracy, dataset
