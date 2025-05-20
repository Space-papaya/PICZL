

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

        bias_z = np.abs(labels - modes)
        frac_z = bias_z/(1+labels)

        outlier_frac = np.where(frac_z >0.15)[0].shape[0]/len(labels)
        accuracy = 1.48 * np.median(frac_z)

        return outlier_frac, accuracy





def ensemble_pdfs(weights, all_pdfs, samples):

	# Compute the weighted sum of pdf scores
	ens_pdf_scores = np.sum([weights[i] * all_pdfs[i] for i in range(len(weights))], axis=0)

	# Normalize each PDF using trapezoidal integration
	areas = np.trapz(ens_pdf_scores, x=samples[1], axis=1)
	norm_ens_pdf_scores = ens_pdf_scores / areas[:, np.newaxis]
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

	#Compute FLASH likelihood for redshift slice [0.4, 1]
	lower = 0.4
	upper = 1.0
	mask = (samples[1] >= lower) & (samples[1] <= upper)
	area_in_interval = np.trapz(norm_ens_pdf_scores[:, mask], x=samples[1][mask], axis=1)


	return norm_ens_pdf_scores, ens_modes, lower_bound_1sig, upper_bound_1sig, lower_bound_3sig, upper_bound_3sig, area_in_interval












