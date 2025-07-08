"""
pdf_distributions.py

Functions for generating photometric redshift PDFs from model predictions.

This module contains utilities to convert model outputs representing
Gaussian mixture parameters into normalized probability distribution functions (PDFs)
over redshift samples. It supports TensorFlow Probability distributions and
computes statistics such as PDF modes and integrals over specified redshift intervals.

"""

import tensorflow as tf
from tqdm import tqdm
from scipy.stats import norm
import tensorflow_probability as tfp
import numpy as np
from scipy.interpolate import interp1d
import sys
from scipy.signal import find_peaks, argrelextrema
from scipy.integrate import cumulative_trapezoid as cumtrapz
import time



def get_pdfs(predictions, num_objects, num_samples):
    """
    Given model output parameters for a Gaussian mixture model (means, stds, and weights),
    this function computes the PDF values for each object over a defined redshift sampling grid.

    Args:
        predictions (np.ndarray or tf.Tensor): Array of shape (num_objects, 3 * num_gaussians),
            containing concatenated means, standard deviations, and weights for each Gaussian component.
        num_objects (int): Number of objects (sources) in the batch.
        num_samples (int): Number of redshift samples to evaluate the PDFs on, spanning [0, 8].

    Returns:
        tuple:
            - pdf_scores (tf.Tensor): Tensor of shape (num_objects, num_samples) with the evaluated PDF values.
            - samples (np.ndarray): Array of shape (num_objects, num_samples) with the redshift sampling points.
    """

    samples = np.array([np.linspace(0, 8, num_samples) for _ in range(num_objects)])

    num_objects, num_gaussians = predictions.shape[0], predictions.shape[1] // 3
    means = predictions[:, :num_gaussians]
    stds = predictions[:, num_gaussians:2 * num_gaussians]
    weights = predictions[:, 2 * num_gaussians:]

    means = means[:, tf.newaxis, :]
    stds = stds[:, tf.newaxis, :]
    weights = weights[:, tf.newaxis, :]

    dists = tfp.distributions.Normal(loc=means, scale=stds)
    pdf_scores = dists.prob(samples[:, :, np.newaxis])
    pdf_scores = tf.reduce_sum(pdf_scores * weights, axis=-1)

    return pdf_scores, samples



def ensemble_pdfs(weights, all_pdfs, samples):
    """
    This function calculates the weighted sum of several PDFs, normalizes the resulting ensemble PDF,
    determines the mode (peak) of the ensemble, and computes the integrated probability within a
    specified redshift slice (default [0.4, 1.0]).

    Args:
        weights (array-like): 1D array of weights corresponding to each PDF, representing model confidence.
        all_pdfs (list or array-like): List or array of shape (num_models, num_objects, num_samples)
            containing individual PDFs from each model.
        samples (np.ndarray): Array of shape (num_objects, num_samples) representing redshift sampling points.

    Returns:
        tuple:
            - norm_ens_pdf_scores (np.ndarray): Normalized ensemble PDFs with shape (num_objects, num_samples).
            - ens_modes (np.ndarray): Array of shape (num_objects,) containing the redshift mode for each ensemble PDF.
            - area_in_interval (np.ndarray): Array of shape (num_objects,) with integrated PDF probabilities
              within the redshift slice [0.4, 1.0].
    """

    ens_pdf_scores = np.sum([weights[i] * all_pdfs[i] for i in range(len(weights))], axis=0)

    areas = np.trapz(ens_pdf_scores, x=samples[1], axis=1)
    norm_ens_pdf_scores = ens_pdf_scores / areas[:, np.newaxis]
    ens_modes = samples[1][np.argmax(norm_ens_pdf_scores, axis=1)]

    lower, upper = 0.4, 1.0
    mask = (samples[1] >= lower) & (samples[1] <= upper)
    area_in_interval = np.trapz(norm_ens_pdf_scores[:, mask], x=samples[1][mask], axis=1)

    return norm_ens_pdf_scores, ens_modes, area_in_interval



def calculate_metrics(ens_modes, labels):
    """
    Calculate photometric redshift performance metrics: outlier fraction and accuracy.

    Args:
        modes (np.ndarray): Estimated redshift modes from PDFs, shape (N,).
        labels (np.ndarray): True (spectroscopic) redshifts, shape (N,).

    Returns:
        outlier_frac (float): Fraction of catastrophic outliers, defined as
            objects where (|z_true - z_pred| / (1 + z_true)) > 0.15.
        accuracy (float): Robust scatter estimate, defined as
            1.48 times the median absolute normalized deviation.
    """

    outlier_frac = np.sum((np.abs(labels - modes) / (1 + labels)) > 0.15) / len(labels)
    accuracy = 1.48 * np.median(np.abs(labels - modes) / (1 + labels))

    return outlier_frac, accuracy



def classify_pdf(z, pdf, hpd_mass=0.68, prominence_threshold=0.15):
    """
    Classify the degeneracy of a photometric redshift PDF and compute the highest posterior density (HPD) interval.

    This function identifies peaks and valleys, and classifies the PDF degeneracy based on
    prominence and separation of secondary peaks relative to the primary peak. It also calculates the HPD interval
    containing a specified mass of the PDF probability.

    Args:
        z (np.ndarray): Array of redshift sample points, assumed sorted and uniformly spaced.
        pdf (np.ndarray): Probability density function values corresponding to z.
        hpd_mass (float, optional): The target cumulative mass for the HPD interval (default: 0.68).
        prominence_threshold (float, optional): Threshold fraction of primary peak height to consider secondary peaks
            significant (default: 0.15).

    Returns:
        dict: Dictionary containing:
            - 'HPD' (tuple): (min, max) redshift bounds of the HPD interval.
            - 'degeneracy' (str): Classification of PDF degeneracy, one of ['none', 'light', 'medium', 'strong'].
            - 'secondary_peaks' (int): Number of secondary peaks passing the prominence filter.
            - 'best_interval' (tuple): Shortest interval around the primary peak containing ≥ hpd_mass probability.

    Notes:
        - The degeneracy classification is based on the number and prominence of secondary peaks relative to the primary peak.
        - The HPD interval is computed as the narrowest interval containing the specified mass of the PDF.
    """

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
        if i == primary_idx:
            continue
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

    sorted_idx = np.argsort(pdf)[::-1]
    cumulative = np.cumsum(pdf[sorted_idx]) * (z[1] - z[0])
    hpd_mask = np.zeros_like(pdf, dtype=bool)
    hpd_mask[sorted_idx[:np.searchsorted(cumulative, hpd_mass) + 1]] = True

    hpd_z = z[hpd_mask]
    hpd_bounds = (np.min(hpd_z), np.max(hpd_z))

    in_hpd = (z >= hpd_bounds[0]) & (z <= hpd_bounds[1])
    z_lim, pdf_lim = z[in_hpd], pdf[in_hpd]
    cdf = np.insert(cumtrapz(pdf_lim, z_lim), 0, 0)

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
        'HPD': hpd_bounds,
        'degeneracy': degeneracy,
        'secondary_peaks': len(filtered_peaks),
        'best_interval': best_int
    }



def batch_classify(z, pdfs, hpd_mass=0.68, prominence_threshold=0.15):
    """
    This function applies `classify_pdf` to each PDF in the input list, reporting progress with a tqdm progress bar,
    and prints the total processing time.

    Args:
        z (np.ndarray): Array of redshift sample points (1D).
        pdfs (List[np.ndarray]): List or array of PDF arrays corresponding to each object.
        hpd_mass (float, optional): Target cumulative probability mass for the HPD interval (default: 0.68).
        prominence_threshold (float, optional): Threshold fraction of primary peak height for peak prominence (default: 0.15).

    Returns:
        List[dict]: List of classification results for each PDF. Each dict contains keys:
            - 'HPD', 'degeneracy', 'secondary_peaks', 'best_interval'
    """

    start = time.time()
    results = [classify_pdf(z, pdf, hpd_mass, prominence_threshold) for pdf in tqdm(pdfs, desc="Classifying PDFs")]
    elapsed = time.time() - start
    print(f"Processed {len(pdfs)} PDFs in {elapsed:.2f} seconds.")

    return results
