"""
Module: fetch_inputs.py

This module provides utility functions to load photometric and imaging data.
It handles catalog ingestion, image loading, and subsampling.

Dependencies:
    - astropy
    - pandas
    - numpy
    - pickle
    - os, sys

Typical usage example:
    catalog, images = fetch_all_inputs("path/to/catalog.fits", "path/to/images/", psf=False, sub_sample_yesno=True, sub_sample_size=max_sources)
"""

from astropy.table import Table
import pickle
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from piczl.config.required_columns import REQUIRED_COLUMNS


def fetch_all_data(
    url_catalog, url_images, exec, psf=False, sub_sample_yesno=True, sub_sample_size=20
):
    """
    Fetches the input catalog and associated images.

    Args:
        url_catalog (str): Path to the FITS catalog.
        url_images (str): Path to the directory containing image numpy files.
        psf (bool): If True, load PSF images.
        sub_sample_yesno (bool): If True, return only the first `sub_sample_size` entries.
        sub_sample_size (int): Number of samples to return if subsampling is enabled.

    Returns:
        tuple: (DataFrame, dict) where the DataFrame is the catalog and the dict contains image arrays.
    """

    dataset = fetch_catalog(url_catalog, exec)
    image_data = fetch_images(url_images, psf)

    if sub_sample_yesno:
        sampled_df = dataset.iloc[:sub_sample_size].reset_index(drop=True)

        for key in image_data:
            image_data[key] = image_data[key][:sub_sample_size]

        return sampled_df, image_data

    return dataset, image_data


def fetch_catalog(url_catalog, execute):
    """
    Reads in the FITS catalog and reorders its columns to match required input format.

    Args:
        url_catalog (str): Path to the FITS catalog.

    Returns:
        pandas.DataFrame: Catalog with columns reordered.
    """

    print("\n >> Processing dataset ...")
    dataset = Table.read(url_catalog).to_pandas()

    REQUIRED_COLUMNS_RUN = [col for col in REQUIRED_COLUMNS if col != "z"]

    if execute == "train":
        required_cols = REQUIRED_COLUMNS
    elif execute == "run":
        required_cols = REQUIRED_COLUMNS_RUN
    else:
        raise ValueError("exec must be either 'run' or 'train'")

    missing_cols = [col for col in required_cols if col not in dataset.columns]
    if missing_cols:
        raise ValueError(
            f"The following required columns are missing from the catalog: {missing_cols}"
        )

    return dataset.reindex(columns=required_cols)


def fetch_images(url_images, psf):
    """
    Loads image datasets for optical and IR bands from NumPy files.

    Args:
        url_images (str): Directory path containing .npy files.
        psf (bool): If True, load PSF image cutouts.

    Returns:
        dict: Dictionary with keys corresponding to image or flux type and values as numpy arrays.
    """

    print(" >> Loading images ...")
    image_data = {}

    def safe_load(filename):
        path = os.path.join(url_images, filename)
        try:
            return np.load(path, allow_pickle=True)
        except FileNotFoundError:
            print(f"[Missing]: '{filename}' not found.")
        except Exception as e:
            print(f"[Error]: Could not load '{filename}' — {e}")
        return None

    images_griz = safe_load("processed_dered_griz_images.npy")
    if images_griz is not None:
        band_names = ["g", "r", "i", "z"]
        image_data.update(
            {f"im_{band}": images_griz[idx] for idx, band in enumerate(band_names)}
        )

    images_griz_col = safe_load("processed_dered_griz_colours.npy")
    if images_griz_col is not None:
        color_bands = ["gr", "gi", "gz", "ri", "rz", "iz"]
        image_data.update(
            {
                f"im_{color_bands[idx]}_col": images_griz_col[idx]
                for idx in range(len(color_bands))
            }
        )

    ap_ims_LS10 = safe_load("aperture_images_LS10.npy")
    if ap_ims_LS10 is not None:
        for band in ["g", "r", "i", "z"]:
            image_data[f"ap_im_{band}"] = ap_ims_LS10.item().get(band)

    ap_ims_WISE = safe_load("aperture_images_WISE.npy")
    if ap_ims_WISE is not None:
        for w in range(1, 5):
            image_data[f"ap_im_w{w}"] = ap_ims_WISE.item().get(f"w{w}")

    ap_ims_LS10_cols = safe_load("aperture_images_LS10_colours.npy")
    if ap_ims_LS10_cols is not None:
        color_bands = ["gr", "gi", "gz", "ri", "rz", "iz"]
        image_data.update(
            {
                f"ap_im_{color_bands[idx]}_col": ap_ims_LS10_cols[idx]
                for idx in range(len(color_bands))
            }
        )

    ap_ims_WISE_cols = safe_load("aperture_images_WISE_colours.npy")
    if ap_ims_WISE_cols is not None:
        wise_color_bands = ["w12", "w13", "w14", "w23", "w24", "w34"]
        image_data.update(
            {
                f"ap_im_{wise_color_bands[idx]}_col": ap_ims_WISE_cols[idx]
                for idx in range(len(wise_color_bands))
            }
        )

    LS10_griz_res = safe_load("dered_griz_residuals.npy")
    if LS10_griz_res is not None:
        for idx, band in enumerate(["g", "r", "i", "z"]):
            image_data[f"res_{band}"] = LS10_griz_res[idx]

    ap_ims_WISE_res = safe_load("aperture_images_WISE_residuals.npy")
    if ap_ims_WISE_res is not None:
        for w in range(1, 5):
            image_data[f"ap_im_w{w}_res"] = ap_ims_WISE_res.item().get(f"w{w}")

    ap_ims_LS10_ivar = safe_load("aperture_images_LS10_ivar.npy")
    if ap_ims_LS10_ivar is not None:
        for band in ["g", "r", "i", "z"]:
            image_data[f"ivar_{band}"] = ap_ims_LS10_ivar.item().get(band)

    ap_ims_WISE_ivar = safe_load("aperture_images_WISE_ivar.npy")
    if ap_ims_WISE_ivar is not None:
        for w in range(1, 5):
            image_data[f"ap_im_w{w}_ivar"] = ap_ims_WISE_ivar.item().get(f"w{w}")

    mod_griz = safe_load("dered_griz_models.npy")
    if mod_griz is not None:
        for idx, band in enumerate(["g", "r", "i", "z"]):
            image_data[f"mod_{band}"] = mod_griz[idx]

        for b1, b2 in [
            ("g", "r"),
            ("g", "i"),
            ("g", "z"),
            ("r", "i"),
            ("r", "z"),
            ("i", "z"),
        ]:
            if f"mod_{b1}" in image_data and f"mod_{b2}" in image_data:
                image_data[f"mod_{b1}{b2}_col"] = np.nan_to_num(
                    np.divide(
                        image_data[f"mod_{b1}"],
                        image_data[f"mod_{b2}"],
                        out=np.zeros_like(image_data[f"mod_{b1}"]),
                        where=image_data[f"mod_{b2}"] != 0,
                    )
                )

    if psf:
        psf_im = safe_load("psf_images.npy")
        if psf_im is not None:
            for idx, band in enumerate(["g", "r", "i", "z"]):
                image_data[f"psf_{band}"] = psf_im[idx]

    print(f" >> Loaded {len(image_data)} image data arrays ...")
    return image_data
