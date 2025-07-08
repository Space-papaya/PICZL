"""
feature_extraction.py

Module to extract, clean, and normalize photometric features from LS10.
The main function prepares numerical, non-spatially dependent scalar features and aperture arrays
for photometry, inverse variance, residuals, and colours, split by active/inactive modes.

Functions:
----------
- grab_features(dataset, mode): Extracts and normalizes subsets of features based on the mode.
"""

import numpy as np
import pandas as pd


def grab_features(dataset, mode):
    """
    This function removes non-relevant features and splits the dataset into
    subsets of spatially independent scalar features and aperture colour arrays
    for LS10 and WISE bands. The features are normalized and scaled to [0,1]
    range for consistent model input.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Input catalogue with photometric and other features.

    mode : str
        Mode of operation. If 'active', a specific subset of features is selected,
        otherwise a different set is used.

    Returns
    -------
    numerical_features : numpy.ndarray
        Array containing all normalized, non-spatially dependent features combined.

    index : pandas.Index
        Original index of the input dataset, useful for aligning results later.
    """

    features = dataset.copy()
    index = features.index
    features = features.drop(["FULLID", "RA", "DEC", "Cat", "type", "TS_ID"], axis=1)

    if mode != "active":
        features_dchisq = np.array(features.iloc[:, 0:5])
        features_snr = np.array(features.iloc[:, [5, 17, 29, 41, 53, 71, 89, 107]])
        features_dered_flux = np.array(features.iloc[:, [6, 18, 30, 42]])
        features_frac_flux = np.array(features.iloc[:, 133:141])
        features_psf_size = np.array(features.iloc[:, 141:145])
        features_shape_e1 = np.array(features.iloc[:, 145])
        features_shape_e1_ivar = np.array(features.iloc[:, 146])
        features_shape_e2 = np.array(features.iloc[:, 147])
        features_shape_e2_ivar = np.array(features.iloc[:, 148])
        features_type = np.array(features.iloc[:, 151:156])
        features_col = np.array(features.iloc[:, 213:229])

        feature_arrays = [
            "features_dchisq",
            "features_snr",
            "features_dered_flux",
            "features_frac_flux",
            "features_psf_size",
            "features_shape_e1",
            "features_shape_e1_ivar",
            "features_shape_e2",
            "features_shape_e2_ivar",
        ]
    else:
        features_dchisq = np.array(features.iloc[:, 0:5])
        features_snr = np.array(features.iloc[:, [5, 17, 29, 41, 53, 71, 89, 107]])
        features_dered_flux = np.array(features.iloc[:, [6, 18, 30, 42]])
        features_frac_flux = np.array(features.iloc[:, 133:141])
        features_type = np.array(features.iloc[:, 151:156])
        features_col = np.array(features.iloc[:, 213:235])

        feature_arrays = [
            "features_dchisq",
            "features_snr",
            "features_dered_flux",
            "features_frac_flux",
        ]

    scaled_features = {}

    for feature in feature_arrays:
        feature_name = feature.replace("features_", "")
        feature_data = np.array(eval(feature))
        global_mean = np.mean(feature_data)
        global_std = np.std(feature_data)

        scaled = (feature_data - global_mean) / global_std
        normalized = (scaled - np.min(scaled)) / (np.max(scaled) - np.min(scaled))

        if normalized.ndim == 1:
            normalized = normalized.reshape(-1, 1)

        scaled_features[f"scaled_feature_{feature_name}"] = normalized

    numerical_features = np.concatenate(
        list(scaled_features.values()) + [features_col, features_type], axis=1
    )

    print(" >> Feature extraction completed")

    return numerical_features, index
