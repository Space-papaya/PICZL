
#Import libraries
from astropy.table import Table
import pickle
import numpy as np
import sys



# ---------------------------------------------------------------
# 2. Import data, change respective paths
# ---------------------------------------------------------------



def fetch_all_inputs(url_catalog, url_images, sub_sample_yesno=True, sub_sample_size=20):


	# Fetch dataset
	dataset = fetch_catalog(url_catalog)

	# Fetch images as a dictionary
	image_data = fetch_images(url_images)

	# Apply downselection if needed
	if sub_sample_yesno:
		sampled_df = dataset.iloc[:sub_sample_size].reset_index(drop=True)

		# Downselect all image arrays
		for key in image_data:
			image_data[key] = image_data[key][:sub_sample_size]

		return sampled_df, image_data

	return dataset, image_data



# ---------------------------------------------------------------
# ---------------------------------------------------------------



def fetch_catalog(url):

	print('\n >> Processing dataset ...')
	dataset = Table.read(url).to_pandas()


	# Load ordered column names from text file
	path = "/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/required_columns.txt"
	with open(path, "r") as f:
		ordered_columns = [line.strip() for line in f.readlines()]

	# Reorder dataset based on saved column order
	dataset = dataset.reindex(columns=ordered_columns)


	return dataset



# ---------------------------------------------------------------
# ---------------------------------------------------------------




def fetch_images(url):
	"""Loads multiple image datasets from NumPy files and returns them as a dictionary."""

	print(" >> Loading images ...")
	image_data = {}

	try:
		# Dered and processed flux cutouts (optical)
		images_griz = np.load(url + "clean_dered_griz.npy")
		band_names = ["g", "r", "i", "z"]
		image_data.update({f"im_{band}": images_griz[idx] for idx, band in enumerate(band_names)})

		# Processed color cutouts (optical)
		images_griz_col = np.load(url + "clean_griz_cols.npy")
		color_bands = ["gr", "gi", "gz", "ri", "rz", "iz"]
		image_data.update({f"im_{color_bands[idx]}_col": images_griz_col[idx] for idx in range(len(color_bands))})

		# Dered aperture flux (optical)
		ap_ims_LS10 = np.load(url + "ap_im_LS10.npy", allow_pickle=True).item()
		image_data.update({f"ap_im_{band}": ap_ims_LS10[band] for band in band_names})

		# Dered aperture flux (IR)
		ap_ims_WISE = np.load(url + "ap_im_WISE.npy", allow_pickle=True).item()
		for w in range(1, 5):
			image_data[f"ap_im_w{w}"] = ap_ims_WISE[f"w{w}"]

		# Dered aperture flux colors (optical)
		ap_ims_LS10_cols = np.load(url + "ap_ims_LS10_cols.npy")
		image_data.update({f"ap_im_{color_bands[idx]}_col": ap_ims_LS10_cols[idx] for idx in range(len(color_bands))})

		# Dered aperture flux colors (IR)
		ap_ims_WISE_cols = np.load(url + "ap_ims_WISE_cols.npy")
		wise_color_bands = ["w12", "w13", "w14", "w23", "w24", "w34"]
		image_data.update({f"ap_im_{wise_color_bands[idx]}_col": ap_ims_WISE_cols[idx] for idx in range(len(wise_color_bands))})

		# Dered flux residuals (optical)
		LS10_griz_res = np.load(url + "dered_griz_resid.npy")
		image_data.update({f"res_{band}": LS10_griz_res[idx] for idx, band in enumerate(band_names)})

		# Dered aperture flux residuals (IR)
		ap_ims_WISE_res = np.load(url + "ap_im_WISE_res.npy", allow_pickle=True).item()
		for w in range(1, 5):
			image_data[f"ap_im_w{w}_res"] = ap_ims_WISE_res[f"w{w}"]

		# Aperture flux inverse variance (optical)
		ap_ims_LS10_ivar = np.load(url + "ap_im_LS10_ivar.npy", allow_pickle=True).item()
		image_data.update({f"ivar_{band}": ap_ims_LS10_ivar[band] for band in band_names})

		# Aperture flux inverse variance (IR)
		ap_ims_WISE_ivar = np.load(url + "ap_im_WISE_ivar.npy", allow_pickle=True).item()
		for w in range(1, 5):
			image_data[f"ap_im_w{w}_ivar"] = ap_ims_WISE_ivar[f"w{w}"]

		# Dered model cutouts (optical)
		mod_griz = np.load(url + "dered_griz_model.npy")
		image_data.update({f"mod_{band}": mod_griz[idx] for idx, band in enumerate(band_names)})

		# Compute model color cutouts (optical)
		for (b1, b2) in [("g", "r"), ("g", "i"), ("g", "z"), ("r", "i"), ("r", "z"), ("i", "z")]:
			image_data[f"mod_{b1}{b2}_col"] = np.nan_to_num(np.divide(image_data[f"mod_{b1}"], image_data[f"mod_{b2}"],
			out=np.zeros_like(image_data[f"mod_{b1}"]), where=(image_data[f"mod_{b2}"] != 0)))

		# PSF images
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

'''

def fetch_images(url):
	"""Loads multiple image datasets from NumPy files and returns them as a dictionary."""
	print(" >> Loading images ...")
	image_data = {}
	failed_files = []

	def safe_load(file_path, allow_pickle=False, as_item=False):
		try:
			data = np.load(file_path, allow_pickle=allow_pickle)
			if as_item:
				return data.item()
			return data
		except FileNotFoundError:
			print(f"File not found: {file_path}")
			failed_files.append(file_path)
			return None
		except Exception as e:
			print(f"Error loading {file_path}: {e}")
			failed_files.append(file_path)
			return None

	# 1. Dered and processed flux cutouts (optical)
	images_griz = safe_load(url + "clean_dered_griz.npy")
	if images_griz is not None:
		band_names = ["g", "r", "i", "z"]
		image_data.update({f"im_{band}": images_griz[idx] for idx, band in enumerate(band_names)})

	# 2. Processed color cutouts (optical)
	images_griz_col = safe_load(url + "clean_griz_cols.npy")
	if images_griz_col is not None:
		color_bands = ["gr", "gi", "gz", "ri", "rz", "iz"]
		image_data.update({f"im_{color_bands[idx]}_col": images_griz_col[idx] for idx in range(len(color_bands))})

	# 3. Dered aperture flux (optical)
	ap_ims_LS10 = safe_load(url + "ap_im_LS10.npy", allow_pickle=True, as_item=True)
	if ap_ims_LS10 is not None:
		image_data.update({f"ap_im_{band}": ap_ims_LS10[band] for band in band_names})

	# 4. Dered aperture flux (IR)
	ap_ims_WISE = safe_load(url + "ap_im_WISE.npy", allow_pickle=True, as_item=True)
	if ap_ims_WISE is not None:
		for w in range(1, 5):
			image_data[f"ap_im_w{w}"] = ap_ims_WISE[f"w{w}"]

	# 5. Dered aperture flux colors (optical)
	ap_ims_LS10_cols = safe_load(url + "ap_ims_LS10_cols.npy")
	if ap_ims_LS10_cols is not None:
		image_data.update({f"ap_im_{color_bands[idx]}_col": ap_ims_LS10_cols[idx] for idx in range(len(color_bands))})

	# 6. Dered aperture flux colors (IR)
	ap_ims_WISE_cols = safe_load(url + "ap_ims_WISE_cols.npy")
	wise_color_bands = ["w12", "w13", "w14", "w23", "w24", "w34"]
	if ap_ims_WISE_cols is not None:
		image_data.update({f"ap_im_{wise_color_bands[idx]}_col": ap_ims_WISE_cols[idx] for idx in range(len(wise_color_bands))})

	# 7. Dered flux residuals (optical)
	LS10_griz_res = safe_load(url + "dered_griz_resid.npy")
	if LS10_griz_res is not None:
		image_data.update({f"res_{band}": LS10_griz_res[idx] for idx, band in enumerate(band_names)})

	# 8. Dered aperture flux residuals (IR)
	ap_ims_WISE_res = safe_load(url + "ap_im_WISE_res.npy", allow_pickle=True, as_item=True)
	if ap_ims_WISE_res is not None:
		for w in range(1, 5):
			image_data[f"ap_im_w{w}_res"] = ap_ims_WISE_res[f"w{w}"]

	# 9. Aperture flux inverse variance (optical)
	ap_ims_LS10_ivar = safe_load(url + "ap_im_LS10_ivar.npy", allow_pickle=True, as_item=True)
	if ap_ims_LS10_ivar is not None:
		image_data.update({f"ivar_{band}": ap_ims_LS10_ivar[band] for band in band_names})

	# 10. Aperture flux inverse variance (IR)
	ap_ims_WISE_ivar = safe_load(url + "ap_im_WISE_ivar.npy", allow_pickle=True, as_item=True)
	if ap_ims_WISE_ivar is not None:
		for w in range(1, 5):
			image_data[f"ap_im_w{w}_ivar"] = ap_ims_WISE_ivar[f"w{w}"]

	# 11. PSF images
	psf_im = safe_load(url + "psf_images.npy")
	if psf_im is not None:
		image_data.update({f"psf_{band}": psf_im[idx] for idx, band in enumerate(band_names)})

	# 12. Dered model cutouts (optical)
	LS10_griz_model = safe_load(url + "dered_griz_model.npy")
	if LS10_griz_model is not None:
		image_data.update({f"mod_{band}": LS10_griz_model[idx] for idx, band in enumerate(band_names)})

	# 13. Derived model color cutouts (optical)
	try:
		for (b1, b2) in [("g", "r"), ("g", "i"), ("g", "z"), ("r", "i"), ("r", "z"), ("i", "z")]:
			if f"mod_{b1}" in image_data and f"mod_{b2}" in image_data:
				image_data[f"mod_{b1}{b2}_col"] = np.nan_to_num(
				np.divide(
				image_data[f"mod_{b1}"],
				image_data[f"mod_{b2}"],
				out=np.zeros_like(image_data[f"mod_{b1}"]),
				where=(image_data[f"mod_{b2}"] != 0)
				))
	except Exception as e:
		print(f"Error computing model color cutouts: {e}")

	print(f" >> Loaded {len(image_data)} features.")
	if failed_files:
		print(f"Some files were missing or caused issues: {failed_files}")
	return image_data if image_data else None

'''

# ---------------------------------------------------------------
# ---------------------------------------------------------------


