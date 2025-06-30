


### import the libraries

import tensorflow as tf
from tqdm import tqdm
from scipy.stats import norm
import tensorflow_probability as tfp
import numpy as np
from scipy.interpolate import interp1d
tfd = tfp.distributions
import sys
from scipy.signal import find_peaks, argrelextrema
from scipy.integrate import cumtrapz
import time

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

	'''
	# Convert the PDFs to CDFs
	cdfs = np.cumsum(norm_ens_pdf_scores, axis=1)

	# Define the confidence percentiles (1 and 3 sigma)
	confidence_percentiles = np.array([0.0015, 0.16, 0.84, 0.9985])

	# Find the indices where the CDFs are closest to the target values
	lower_bound_3sig = samples[0][np.abs(cdfs - confidence_percentiles[0]).argmin(axis=1)]
	lower_bound_1sig = samples[0][np.abs(cdfs - confidence_percentiles[1]).argmin(axis=1)]
	upper_bound_1sig = samples[0][np.abs(cdfs - confidence_percentiles[2]).argmin(axis=1)]
	upper_bound_3sig = samples[0][np.abs(cdfs - confidence_percentiles[3]).argmin(axis=1)]
	'''

	#Compute FLASH likelihood for redshift slice [0.4, 1]
	lower = 0.4
	upper = 1.0
	mask = (samples[1] >= lower) & (samples[1] <= upper)
	area_in_interval = np.trapz(norm_ens_pdf_scores[:, mask], x=samples[1][mask], axis=1)

	return norm_ens_pdf_scores, ens_modes, area_in_interval




def get_point_estimates(pdf_scores, samples):

        # Compute the weighted sum of pdf scores
        norm_pdf_scores = pdf_scores / np.sum(pdf_scores, axis=1, keepdims=True)
        modes = samples[1][np.argmax(norm_pdf_scores, axis=1)]

        return modes




def classify_pdf(z, pdf, hpd_mass=0.68, prominence_threshold=0.15):
	pdf = pdf / np.trapz(pdf, z)
	peak_idxs, _ = find_peaks(pdf)
	valleys = argrelextrema(pdf, np.less)[0]

	primary_idx = np.argmax(pdf)
	z_peak = z[primary_idx]
	p_max = pdf[primary_idx]

	height_cut = 0.01 * p_max
	candidate_peaks = [i for i in peak_idxs if pdf[i] >= height_cut]
	filtered_peaks = []

	for i in candidate_peaks:
		if i == primary_idx: continue
		if i < primary_idx:
			drop = pdf[i] - np.min(pdf[i:primary_idx+1])
			if drop > prominence_threshold * p_max:
				filtered_peaks.append(i)
		else:
			v_idx = valleys[valleys < i][-1] if np.any(valleys < i) else 0
			rise = pdf[i] - pdf[v_idx]
			if rise > prominence_threshold * p_max:
				filtered_peaks.append(i)

	if len(candidate_peaks) <= 1:
		degeneracy = 'none'
	elif len(filtered_peaks) == 0:
		degeneracy = 'light'
	else:
		max_sep = max(abs(z[i] - z_peak) for i in filtered_peaks)
		if max_sep > 0.15 * (1 + z_peak):
			degeneracy = 'strong'
		else:
			degeneracy = 'medium'

	# Compute HPD mask
	sorted_idx = np.argsort(pdf)[::-1]
	cumulative = np.cumsum(pdf[sorted_idx]) * (z[1] - z[0])
	hpd_mask = np.zeros_like(pdf, dtype=bool)
	hpd_mask[sorted_idx[:np.searchsorted(cumulative, hpd_mass) + 1]] = True

	hpd_z = z[hpd_mask]
	hpd_bounds = (np.min(hpd_z), np.max(hpd_z))

	# Trim to HPD interval
	in_hpd = (z >= hpd_bounds[0]) & (z <= hpd_bounds[1])
	z_lim, pdf_lim = z[in_hpd], pdf[in_hpd]
	cdf = np.insert(cumtrapz(pdf_lim, z_lim), 0, 0)

	# Find shortest window with mass ≥ hpd_mass containing peak
	min_width = np.inf
	best_int = (z_lim[0], z_lim[-1])
	for i in range(len(z_lim)):
		j = np.searchsorted(cdf, cdf[i] + hpd_mass)
		if j < len(z_lim) and z_lim[i] <= z_peak <= z_lim[j]:
			width = z_lim[j] - z_lim[i]
			if width < min_width:
				min_width = width
				best_int = (z_lim[i], z_lim[j])



	return {
		'z_peak': z_peak,
		'HPD': hpd_bounds,
		'degeneracy': degeneracy,
		'secondary_peaks': len(filtered_peaks),
		'best_interval': best_int
		}




def batch_classify(z, pdfs, hpd_mass=0.68, prominence_threshold=0.15):
	start = time.time()
	results = [classify_pdf(z, pdf, hpd_mass, prominence_threshold) for pdf in tqdm(pdfs, desc="Classifying PDFs")]
	elapsed = time.time() - start
	print(f"Processed {len(pdfs)} PDFs in {elapsed:.2f} seconds.")
	return results



