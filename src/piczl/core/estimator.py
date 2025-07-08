import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from piczl.config.default_models import CONFIGS
from piczl.utilities import *


def run_estimation(
    catalog_path, image_path, mode, sub_sample, max_sources, pdf_samples=4001
):
	"""
	Run photometric redshift estimation on a catalog of LS10 galaxies/AGN.

	This function loads data, preprocesses it, extracts features, loads ensemble models,
	predicts redshift PDFs, ensembles the predictions, and classifies PDF degeneracies.

	Parameters
	----------
	catalog_path : str
	Path to the input catalog file.
	image_path : str
	Path to the corresponding image data.
	mode : str
	Processing mode; must be 'active' or 'inactive'.
	sub_sample : bool
	Whether to use a subsample of sources for testing.
	max_sources : int
	Maximum number of sources to process (only if sub_sample is True).
	pdf_samples : int, optional
	Number of redshift samples in the PDF (default is 4001).

	Returns
	-------
	z_modes : np.ndarray
	Array of mode redshift values for each object.
	l1s : list of float
	Lower bounds of the best HPD intervals per object.
	u1s : list of float
	Upper bounds of the best HPD intervals per object.
	degeneracy : list of str
	Degeneracy classification labels for each PDF.
	dataset : pandas.DataFrame
	The preprocessed catalog dataset.
	"""

	psf = False if mode == "active" else True

	dataset, image_data = fetch_inputs.fetch_all_data(
	    catalog_path,
	    image_path,
	    exec="run",
	    psf=psf,
	    sub_sample_yesno=sub_sample,
	    sub_sample_size=max_sources,
	)
	dataset = cat_preprocessing.run_all_preprocessing(dataset)
	features, index = feature_extraction.grab_features(dataset, mode)
	images, images_col = image_processing.stack_images(image_data)

	config = CONFIGS[mode]
	model_files = config["model_files"]
	weights = np.array(config["model_weights"])
	normalized_weights = weights / np.sum(weights)

	print(" >> Running models to estimate redshifts ...")
	print("\n")

	all_pdfs = []
	for model_file in model_files:
	    model_path = model_loader.get_model_path(mode, model_file)
	    model = load_model(str(model_path), compile=False)
	    preds = model.predict([images, images_col, features])
	    pdfs, samples = pdf_distributions.get_pdfs(preds, len(dataset), pdf_samples)
	    all_pdfs.append(pdfs)

	norm_ens_pdfs, z_modes, areas = pdf_distributions.ensemble_pdfs(
	    normalized_weights, all_pdfs, samples
	)
	results = pdf_distributions.batch_classify(samples[0], norm_ens_pdfs)

	l1s = [round(res["best_interval"][0], 3) for res in results]
	u1s = [round(res["best_interval"][1], 3) for res in results]
	degeneracy = [res["degeneracy"] for res in results]

	return z_modes, l1s, u1s, degeneracy, dataset
