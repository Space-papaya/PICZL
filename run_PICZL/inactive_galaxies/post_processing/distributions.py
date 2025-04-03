

### import the libraries

import tensorflow as tf
from tqdm import tqdm
from scipy.stats import norm
import tensorflow_probability as tfp
import numpy as np
from scipy.interpolate import interp1d
tfd = tfp.distributions



##################################


def get_pdfs(predictions, num_objects, num_samples):

	### set range for sampling
	samples = np.array([np.linspace(0, 8, num_samples) for _ in range(num_objects)])

	### dissect prediction paramaters
	num_objects, num_gaussians = predictions.shape[0], predictions.shape[1] // 3
	means = predictions[:, :num_gaussians]
	stds = predictions[:, num_gaussians:2 * num_gaussians]
	weights = predictions[:, 2 * num_gaussians:]

	# Expand dimensions to align shapes for broadcasting
	means = means[:, tf.newaxis, :]
	stds = stds[:, tf.newaxis, :]
	weights = weights[:, tf.newaxis, :]

	# --------------
	### Deal with PDF
	# --------------

	# Create Normal distributions for all objects and all Gaussians
	dists = tfd.Normal(loc=means, scale=stds)

	# Calculate the PDF scores for all samples and all objects
	pdf_scores = dists.prob(samples[:, :, np.newaxis])

	# Multiply by the weights and sum along the last axis
	pdf_scores = tf.reduce_sum(pdf_scores * weights, axis=-1)


	return pdf_scores, samples





def calculate_metrics(modes, labels):

	abs_bias = np.abs(labels - modes) / (1+labels)
	bias = np.mean((labels - modes) / (1+labels))

	acc = 1.4826 * np.median(abs_bias)
	outlier_frac = np.where(abs_bias >0.15)[0].shape[0]/len(labels)

	return acc, outlier_frac





def ensemble_pdfs(weights, all_pdfs, samples):

	# Compute the weighted sum of pdf scores
	ens_pdf_scores = np.sum([weights[i] * all_pdfs[i] for i in range(len(weights))], axis=0)
	norm_ens_pdf_scores = ens_pdf_scores / np.sum(ens_pdf_scores, axis=1, keepdims=True)

	ens_modes = samples[1][np.argmax(norm_ens_pdf_scores, axis=1)]


	# Convert the PDFs to CDFs
	cdfs = np.cumsum(norm_ens_pdf_scores, axis=1)

	# Define the confidence percentiles (1 and 3 sigma)
	confidence_percentiles = np.array([0.0015, 0.16, 0.84, 0.9985])

	# Find the indices where the CDFs are closest to the target values
	lower_bound_3sig = samples[0][np.abs(cdfs - confidence_percentiles[0]).argmin(axis=1)]
	lower_bound_1sig = samples[0][np.abs(cdfs - confidence_percentiles[1]).argmin(axis=1)]
	upper_bound_1sig = samples[0][np.abs(cdfs - confidence_percentiles[2]).argmin(axis=1)]
	upper_bound_3sig = samples[0][np.abs(cdfs - confidence_percentiles[3]).argmin(axis=1)]


	return norm_ens_pdf_scores, ens_modes, lower_bound_1sig, upper_bound_1sig, lower_bound_3sig, upper_bound_3sig












