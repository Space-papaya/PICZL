
#Import libraries
from astropy.table import Table
import pickle
import numpy as np
import sys



# ---------------------------------------------------------------
# 2. Import data, change respective paths
# ---------------------------------------------------------------



def fetch_all_inputs(url_catalog, url_images, sub_sample_yesno, sub_sample_size, url_catalog_og):


	# Fetch dataset
	dataset = fetch_catalog(url_catalog, url_catalog_og)

	# Fetch images as a dictionary to keep things organized
	image_keys = [
	"g", "r", "i", "z", "gr", "gi", "gz", "ri", "rz", "iz",
	"mod_g", "mod_r", "mod_i", "mod_z", "mod_gr", "mod_gi", "mod_gz",
	"mod_ri", "mod_rz", "mod_iz",
	"ap_ims_g_band_res", "ap_ims_r_band_res", "ap_ims_i_band_res", "ap_ims_z_band_res",
	"ap_ims_g_band_ivar", "ap_ims_r_band_ivar", "ap_ims_i_band_ivar", "ap_ims_z_band_ivar",
	"ap_ims_w1_band_res", "ap_ims_w2_band_res", "ap_ims_w3_band_res", "ap_ims_w4_band_res",
	"ap_ims_w1_band_ivar", "ap_ims_w2_band_ivar", "ap_ims_w3_band_ivar", "ap_ims_w4_band_ivar",
	"ap_ims_g_band", "ap_ims_r_band", "ap_ims_i_band", "ap_ims_z_band",
	"ap_ims_w1_band", "ap_ims_w2_band", "ap_ims_w3_band", "ap_ims_w4_band",
	"ap_ims_gi_band", "ap_ims_gr_band", "ap_ims_gz_band", "ap_ims_ri_band",
	"ap_ims_rz_band", "ap_ims_iz_band", "ap_ims_w12_band", "ap_ims_w13_band",
	"ap_ims_w14_band", "ap_ims_w23_band", "ap_ims_w24_band", "ap_ims_w34_band",
	"psf_g", "psf_r", "psf_i", "psf_z"
	]

	image_data = dict(zip(image_keys, fetch_images(url_images)))

	# Apply downselection if needed
	if sub_sample_yesno:
		sampled_df = dataset.sample(n=sub_sample_size, random_state=42)
		sample_indices = sampled_df.index

		# Downselect all image arrays
		for key in image_data:
			image_data[key] = image_data[key][sample_indices]

		return sampled_df, image_data

	return dataset, image_data



# ---------------------------------------------------------------
# ---------------------------------------------------------------



def fetch_catalog(url, url_og):

	print('\n >> Processing dataset ...')
	dataset = Table.read(url).to_pandas()

	dataset_og = Table.read(url_og).to_pandas()
	dataset=dataset[dataset_og.columns]


	return dataset



# ---------------------------------------------------------------
# ---------------------------------------------------------------




def fetch_images(url):
"""Loads multiple image datasets from NumPy files and returns them as a dictionary."""

	print(" >> Loading images ...")
	image_data = {}

	try:
		# Load cleaned flux band cutouts
		images_griz = np.load(url + "clean_dered_griz.npy")
		band_names = ["g", "r", "i", "z"]
		image_data.update({f"images_{band}": images_griz[idx] for idx, band in enumerate(band_names)})

		# Load cleaned color cutouts
		images_griz_col = np.load(url + "clean_griz_cols.npy")
		color_bands = ["gr", "gi", "gz", "ri", "rz", "iz"]
		image_data.update({f"images_{color_bands[idx]}": images_griz_col[idx] for idx in range(len(color_bands))})

		# Load model flux band cutouts
		mod_griz = np.load(url + "model_dered_griz.npy")
		image_data.update({f"mod_{band}": mod_griz[idx] for idx, band in enumerate(band_names)})

		# Compute model colors
		for (b1, b2) in [("g", "r"), ("g", "i"), ("g", "z"), ("r", "i"), ("r", "z"), ("i", "z")]:
			image_data[f"mod_{b1}{b2}"] = np.nan_to_num(np.divide(image_data[f"mod_{b1}"], image_data[f"mod_{b2}"],
			out=np.zeros_like(image_data[f"mod_{b1}"]), where=(image_data[f"mod_{b2}"] != 0)))

		# Load aperture flux residuals (optical)
		LS10_griz_res = np.load(url + "resid_dered_griz.npy")
		image_data.update({f"ap_ims_{band}_res": LS10_griz_res[idx] for idx, band in enumerate(band_names)})

		# Load aperture flux inverse variance (optical)
		ap_ims_LS10_ivar = np.load(url + "ap_im_LS10_ivar.npy", allow_pickle=True).item()
		image_data.update({f"ap_ims_{band}_ivar": ap_ims_LS10_ivar[band] for band in band_names})

		# Load aperture flux residuals (WISE)
		ap_ims_WISE_res = np.load(url + "ap_im_WISE_res.npy", allow_pickle=True).item()
		for w in range(1, 5):
			image_data[f"ap_ims_w{w}_band_res"] = ap_ims_WISE_res[f"w{w}"]

		# Load aperture flux inverse variance (WISE)
		ap_ims_WISE_ivar = np.load(url + "ap_im_WISE_ivar.npy", allow_pickle=True).item()
		for w in range(1, 5):
			image_data[f"ap_ims_w{w}_band_ivar"] = ap_ims_WISE_ivar[f"w{w}"]


		# Load aperture flux (optical)
		ap_ims_LS10 = np.load(url + "ap_im_LS10.npy", allow_pickle=True).item()
		image_data.update({f"ap_ims_{band}_band": ap_ims_LS10[band] for band in band_names})

		# Load aperture flux (WISE)
		ap_ims_WISE = np.load(url + "ap_im_WISE.npy", allow_pickle=True).item()
		for w in range(1, 5):
			image_data[f"ap_ims_w{w}_band"] = ap_ims_WISE[f"w{w}"]

		# Load aperture flux colors (optical)
		ap_ims_LS10_cols = np.load(url + "ap_ims_LS10_cols.npy")
		image_data.update({f"ap_ims_{color_bands[idx]}_band": ap_ims_LS10_cols[idx] for idx in range(len(color_bands))})

		# Load aperture flux colors (WISE)
		ap_ims_WISE_cols = np.load(url + "ap_ims_WISE_cols.npy")
		wise_color_bands = ["w12", "w13", "w14", "w23", "w24", "w34"]
		image_data.update({f"ap_ims_{wise_color_bands[idx]}_band": ap_ims_WISE_cols[idx] for idx in range(len(wise_color_bands))})

		# Load PSF images
		psf_im = np.load(url + "psf_images.npy")
		image_data.update({f"psf_{band}": psf_im[idx] for idx, band in enumerate(band_names)})


		#56 image features, 60 when PSF is included
		return image_data

		except FileNotFoundError as e:
			print(f"Error: {e}")
		return None
		except Exception as e:
			print(f"Unexpected error: {e}")
		return None




# ---------------------------------------------------------------
# ---------------------------------------------------------------


