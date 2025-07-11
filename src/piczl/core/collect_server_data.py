from astropy.table import Table
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import shutil
import glob
import time
import urllib.request
import io
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import concurrent.futures
from urllib.error import HTTPError
import os
import sys


# Resolve and add path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(module_path)

# Print the resolved path
print(">> Module path added to sys.path:", module_path)

# Now import
from piczl.utilities import *


def get_data(df, type):

	RA = np.array(df['ra'])
	DEC = np.array(df['dec'])
	bands = ['g', 'r', 'i', 'z']  # for LS-DR10 data

	# Initialize a list to store the arrays for each source
	all_sources_array = []

	# Define the appropriate URL and save path based on the type
	if type == 'obs':
		url_base = "https://www.legacysurvey.org/viewer-dev/cutout.fits?ra={ra}&dec={dec}&layer=ls-dr10&pixscale=0.262&size=23&bands=griz"
	elif type == 'model':
		url_base = "https://www.legacysurvey.org/viewer-dev/cutout.fits?ra={ra}&dec={dec}&layer=ls-dr10-model&pixscale=0.262&size=23&bands=griz"
	elif type == 'resid':
		url_base = "https://www.legacysurvey.org/viewer-dev/cutout.fits?ra={ra}&dec={dec}&layer=ls-dr10-resid&pixscale=0.262&size=23&bands=griz"

	for i in tqdm(range(len(df))):
		# Initialize an empty array for each source
		source_array = np.zeros((4, 23, 23))

		# Generate the URL for the current source
		url = url_base.format(ra=RA[i], dec=DEC[i])

		try:
			with urllib.request.urlopen(url) as response:
				data = response.read()
				with fits.open(io.BytesIO(data)) as hdulist:
					image_data_griz = hdulist[0].data  # Assuming the data is in the primary HDU

					# Loop over each band to apply MW correction
					for band_index, band in enumerate(bands):
						# Get the MW transmission data for the current band
						mw = df[f"mw_transmission_{band}"]

						# Correct the data for MW transmission
						source_array[band_index] = image_data_griz[band_index] / mw[i]

			# Append the source_array to the list of all sources
			all_sources_array.append(source_array)

		except Exception as e:
			print(f"Error fetching data for source {i}: {e}")
			continue

	# Convert the list of arrays to a single NumPy array
	all_sources_array = np.transpose(np.array(all_sources_array), (1, 0, 2, 3))


	return all_sources_array


'''
file_path = "/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/random_cats/"
data = Table.read(file_path + 'LS_n2.fits', hdu=1)
dataset = data.to_pandas()


#Get the observations
dered_griz_obs = get_data(dataset, "model")
np.save(file_path + "dered_griz_models.npy", dered_griz_obs)
'''

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from tqdm import tqdm


##################################################################
##################################################################


@jit
# Define pixel_threshold function
def pixel_threshold(image, threshold):
	pixel_noise = np.random.uniform(0.00001, 0.00005)
	return jnp.where(image > threshold, image, pixel_noise)


##################################################################


def clean_obs(images_g, images_r, images_i, images_z, path):


	for i in tqdm(range(0, len(images_g))):
		for j in range(0,23):
			for k in range(0,23):
				#corner and rim
				if j<1: #upper corners
					if k<1: #top left
						pixels_edge_g = images_g[i][j][k+1], images_g[i][j+1][k], images_g[i][j+1][k+1]
						pixels_edge_r = images_r[i][j][k+1], images_r[i][j+1][k], images_r[i][j+1][k+1]
						pixels_edge_i = images_i[i][j][k+1], images_i[i][j+1][k], images_i[i][j+1][k+1]
						pixels_edge_z = images_z[i][j][k+1], images_z[i][j+1][k], images_z[i][j+1][k+1]
					if k>21: #top right
						pixels_edge_g = images_g[i][j][k-1], images_g[i][j+1][k], images_g[i][j+1][k-1]
						pixels_edge_r = images_r[i][j][k-1], images_r[i][j+1][k], images_r[i][j+1][k-1]
						pixels_edge_i = images_i[i][j][k-1], images_i[i][j+1][k], images_i[i][j+1][k-1]
						pixels_edge_z = images_z[i][j][k-1], images_z[i][j+1][k], images_z[i][j+1][k-1]
					if k>0 and k<22: #top side
						pixels_edge_g = images_g[i][j][k-1], images_g[i][j+1][k-1], images_g[i][j+1][k], images_g[i][j+1][k+1], images_g[i][j][k+1]
						pixels_edge_r = images_r[i][j][k-1], images_r[i][j+1][k-1], images_r[i][j+1][k], images_r[i][j+1][k+1], images_r[i][j][k+1]
						pixels_edge_i = images_i[i][j][k-1], images_i[i][j+1][k-1], images_i[i][j+1][k], images_i[i][j+1][k+1], images_i[i][j][k+1]
						pixels_edge_z = images_z[i][j][k-1], images_z[i][j+1][k-1], images_z[i][j+1][k], images_z[i][j+1][k+1], images_z[i][j][k+1]

					if (images_g[i][j][k]/np.mean(pixels_edge_g)) > 3 or (images_g[i][j][k]/np.mean(pixels_edge_g)) < 1/3:
						images_g[i][j][k] = np.mean(pixels_edge_g)
					if (images_r[i][j][k]/np.mean(pixels_edge_r)) > 3 or (images_r[i][j][k]/np.mean(pixels_edge_r)) < 1/3:
						images_r[i][j][k] = np.mean(pixels_edge_r)
					if (images_i[i][j][k]/np.mean(pixels_edge_i)) > 3 or (images_i[i][j][k]/np.mean(pixels_edge_i)) < 1/3:
						images_i[i][j][k] = np.mean(pixels_edge_i)
					if (images_z[i][j][k]/np.mean(pixels_edge_z)) > 3 or (images_z[i][j][k]/np.mean(pixels_edge_z)) < 1/3:
						images_z[i][j][k] = np.mean(pixels_edge_z)




	for i in tqdm(range(0, len(images_g))):
		for j in range(0,23):
			for k in range(0,23):
				#left side right side
				if j>0 and j<22 and (k<1 or k>21):
					if k<1: #left side
						pixels_edge_g = images_g[i][j-1][k], images_g[i][j-1][k+1], images_g[i][j][k+1], images_g[i][j+1][k+1], images_g[i][j+1][k]
						pixels_edge_r = images_r[i][j-1][k], images_r[i][j-1][k+1], images_r[i][j][k+1], images_r[i][j+1][k+1], images_r[i][j+1][k]
						pixels_edge_i = images_i[i][j-1][k], images_i[i][j-1][k+1], images_i[i][j][k+1], images_i[i][j+1][k+1], images_i[i][j+1][k]
						pixels_edge_z = images_z[i][j-1][k], images_z[i][j-1][k+1], images_z[i][j][k+1], images_z[i][j+1][k+1], images_z[i][j+1][k]
					if k>21: #right side
						pixels_edge_g = images_g[i][j-1][k], images_g[i][j-1][k-1], images_g[i][j][k-1], images_g[i][j+1][k-1], images_g[i][j+1][k]
						pixels_edge_r = images_r[i][j-1][k], images_r[i][j-1][k-1], images_r[i][j][k-1], images_r[i][j+1][k-1], images_r[i][j+1][k]
						pixels_edge_i = images_i[i][j-1][k], images_i[i][j-1][k-1], images_i[i][j][k-1], images_i[i][j+1][k-1], images_i[i][j+1][k]
						pixels_edge_z = images_z[i][j-1][k], images_z[i][j-1][k-1], images_z[i][j][k-1], images_z[i][j+1][k-1], images_z[i][j+1][k]


					if (images_g[i][j][k]/np.mean(pixels_edge_g)) > 3 or (images_g[i][j][k]/np.mean(pixels_edge_g)) < 1/3:
						images_g[i][j][k] = np.mean(pixels_edge_g)
					if (images_r[i][j][k]/np.mean(pixels_edge_r)) > 3 or (images_r[i][j][k]/np.mean(pixels_edge_r)) < 1/3:
						images_r[i][j][k] = np.mean(pixels_edge_r)
					if (images_i[i][j][k]/np.mean(pixels_edge_i)) > 3 or (images_i[i][j][k]/np.mean(pixels_edge_i)) < 1/3:
						images_i[i][j][k] = np.mean(pixels_edge_i)
					if (images_z[i][j][k]/np.mean(pixels_edge_z)) > 3 or (images_z[i][j][k]/np.mean(pixels_edge_z)) < 1/3:
						images_z[i][j][k] = np.mean(pixels_edge_z)



	for i in tqdm(range(0, len(images_g))):
		for j in range(1,22):
			for k in range(1,22):
				#main body
				if j>0 and j<22 and k>0 and k<22:
					pixels_g = [images_g[i][j-1][k-1], images_g[i][j-1][k], images_g[i][j-1][k+1], images_g[i][j][k-1], images_g[i][j][k+1], images_g[i][j+1][k-1], images_g[i][j+1][k], images_g[i][j+1][k+1]]  
					pixels_r = [images_r[i][j-1][k-1], images_r[i][j-1][k], images_r[i][j-1][k+1], images_r[i][j][k-1], images_r[i][j][k+1], images_r[i][j+1][k-1], images_r[i][j+1][k], images_r[i][j+1][k+1]]
					pixels_i = [images_i[i][j-1][k-1], images_i[i][j-1][k], images_i[i][j-1][k+1], images_i[i][j][k-1], images_i[i][j][k+1], images_i[i][j+1][k-1], images_i[i][j+1][k], images_i[i][j+1][k+1]]
					pixels_z = [images_z[i][j-1][k-1], images_z[i][j-1][k], images_z[i][j-1][k+1], images_z[i][j][k-1], images_z[i][j][k+1], images_z[i][j+1][k-1], images_z[i][j+1][k], images_z[i][j+1][k+1]]

				if (images_g[i][j][k]/np.median(pixels_g)) > 3 or (images_g[i][j][k]/np.mean(pixels_g)) < 1/3:
					images_g[i][j][k] = np.median(pixels_g)
				if (images_r[i][j][k]/np.median(pixels_r)) > 3 or (images_r[i][j][k]/np.mean(pixels_r)) < 1/3:
					images_r[i][j][k] = np.median(pixels_r)
				if (images_i[i][j][k]/np.median(pixels_i)) > 3 or (images_i[i][j][k]/np.mean(pixels_i)) < 1/3:
					images_i[i][j][k] = np.median(pixels_i)
				if (images_z[i][j][k]/np.median(pixels_z)) > 3 or (images_z[i][j][k]/np.mean(pixels_z)) < 1/3:
					images_z[i][j][k] = np.median(pixels_z)

	#np.save('/home/wroster/scripts/data/diff/new/g_after_noise_b.npy', images_g)


	for i in tqdm(range(0, len(images_g))):
		for j in range(0,23):
			for k in range(0,23):
				if j>21: #lower corners
					if k<1: #lower left
						pixels_edge_g = images_g[i][j][k+1], images_g[i][j-1][k], images_g[i][j-1][k+1]
						pixels_edge_r = images_r[i][j][k+1], images_r[i][j-1][k], images_r[i][j-1][k+1]
						pixels_edge_i = images_i[i][j][k+1], images_i[i][j-1][k], images_i[i][j-1][k+1]
						pixels_edge_z = images_z[i][j][k+1], images_z[i][j-1][k], images_z[i][j-1][k+1]
					if k>21: #lower right
						pixels_edge_g = images_g[i][j][k-1], images_g[i][j-1][k], images_g[i][j-1][k-1]
						pixels_edge_r = images_r[i][j][k-1], images_r[i][j-1][k], images_r[i][j-1][k-1]
						pixels_edge_i = images_i[i][j][k-1], images_i[i][j-1][k], images_i[i][j-1][k-1]
						pixels_edge_z = images_z[i][j][k-1], images_z[i][j-1][k], images_z[i][j-1][k-1]
					if k>0 and k<22: #lower side
						pixels_edge_g = images_g[i][j][k-1], images_g[i][j-1][k-1], images_g[i][j-1][k], images_g[i][j-1][k+1], images_g[i][j][k+1]
						pixels_edge_r = images_r[i][j][k-1], images_r[i][j-1][k-1], images_r[i][j-1][k], images_r[i][j-1][k+1], images_r[i][j][k+1]
						pixels_edge_i = images_i[i][j][k-1], images_i[i][j-1][k-1], images_i[i][j-1][k], images_i[i][j-1][k+1], images_i[i][j][k+1]
						pixels_edge_z = images_z[i][j][k-1], images_z[i][j-1][k-1], images_z[i][j-1][k], images_z[i][j-1][k+1], images_z[i][j][k+1]

					if (images_g[i][j][k]/np.mean(pixels_edge_g)) > 3 or (images_g[i][j][k]/np.mean(pixels_edge_g)) < 1/3:
						images_g[i][j][k] = np.mean(pixels_edge_g)
					if (images_r[i][j][k]/np.mean(pixels_edge_r)) > 3 or (images_r[i][j][k]/np.mean(pixels_edge_r)) < 1/3:
						images_r[i][j][k] = np.mean(pixels_edge_r)
					if (images_i[i][j][k]/np.mean(pixels_edge_i)) > 3 or (images_i[i][j][k]/np.mean(pixels_edge_i)) < 1/3:
						images_i[i][j][k] = np.mean(pixels_edge_i)
					if (images_z[i][j][k]/np.mean(pixels_edge_z)) > 3 or (images_z[i][j][k]/np.mean(pixels_edge_z)) < 1/3:
						images_z[i][j][k] = np.mean(pixels_edge_z)

	#np.save('/home/wroster/scripts/data/diff/new/g_after_noise_a.npy', images_g)


	#For images which only have noise, set them to zero as non-detection
	limit = 0.0001
	default_mag = 0

	if np.max(images_g[i]) < limit:
		for j in range(0,23):
			for k in range(0,23):
				images_g[i][j][k] = default_mag

	if np.max(images_r[i]) < limit:
		for j in range(0,23):
			for k in range(0,23):
				images_r[i][j][k] = default_mag

	if np.max(images_i[i]) < limit:
		for j in range(0,23):
			for k in range(0,23):
				images_i[i][j][k] = default_mag

	if np.max(images_z[i]) < limit:
		for j in range(0,23):
			for k in range(0,23):
				images_z[i][j][k] = default_mag



	clean_griz = np.stack((images_g, images_r, images_i, images_z), axis=0)
	np.save(path + 'processed_dered_images.npy.npy', clean_griz)


##################################################################

def clean_cols(images_g, images_r, images_i, images_z, path):


	images_gr=np.zeros((len(images_g),23,23))
	images_gi=np.zeros((len(images_g),23,23))
	images_gz=np.zeros((len(images_g),23,23))
	images_ri=np.zeros((len(images_g),23,23))
	images_rz=np.zeros((len(images_g),23,23))
	images_iz=np.zeros((len(images_g),23,23))

	default = -99


	for i in tqdm(range(0, len(images_g))):
		#g-r
		if (np.max(images_g[i]) == 0) |  (np.max(images_r[i]) == 0):
			for j in range(0,23):
				for k in range(0,23):
					images_gr[i][j][k] = default
		else:
			for j in range(0,23):
				for k in range(0,23):
					images_gr[i][j][k] = (22.5 - (2.5*(np.log10(images_g[i][j][k])))) - (22.5 - (2.5*(np.log10(images_r[i][j][k]))))


		#g-i
		if (np.max(images_g[i]) == 0) |  (np.max(images_i[i]) == 0):
			for j in range(0,23):
				for k in range(0,23):
					images_gi[i][j][k] = default
		else:
			for j in range(0,23):
				for k in range(0,23):
					images_gi[i][j][k] = (22.5 - (2.5*(np.log10(images_g[i][j][k])))) - (22.5 - (2.5*(np.log10(images_i[i][j][k]))))


		#g-z
		if (np.max(images_g[i]) == 0) |  (np.max(images_z[i]) == 0):
			for j in range(0,23):
				for k in range(0,23):
					images_gz[i][j][k] = default
		else:
			for j in range(0,23):
				for k in range(0,23):
					images_gz[i][j][k] = (22.5 - (2.5*(np.log10(images_g[i][j][k])))) - (22.5 - (2.5*(np.log10(images_z[i][j][k]))))


		#r-i
		if (np.max(images_r[i]) == 0) |  (np.max(images_i[i]) == 0):
			for j in range(0,23):
				for k in range(0,23):
					images_ri[i][j][k] = default
		else:
			for j in range(0,23):
				for k in range(0,23):
					images_ri[i][j][k] = (22.5 - (2.5*(np.log10(images_r[i][j][k])))) - (22.5 - (2.5*(np.log10(images_i[i][j][k]))))


		#r-z
		if (np.max(images_r[i]) == 0) |  (np.max(images_z[i]) == 0):
			for j in range(0,23):
				for k in range(0,23):
					images_rz[i][j][k] = default
		else:
			for j in range(0,23):
				for k in range(0,23):
					images_rz[i][j][k] = (22.5 - (2.5*(np.log10(images_r[i][j][k])))) - (22.5 - (2.5*(np.log10(images_z[i][j][k]))))


		#i-z
		if (np.max(images_i[i]) == 0) |  (np.max(images_z[i]) == 0):
			for j in range(0,23):
				for k in range(0,23):
					images_iz[i][j][k] = default
		else:
			for j in range(0,23):
				for k in range(0,23):
					images_iz[i][j][k] = (22.5 - (2.5*(np.log10(images_i[i][j][k])))) - (22.5 - (2.5*(np.log10(images_z[i][j][k]))))




	for i in tqdm(range(0, len(images_g))):
		for j in range(1,22):
			for k in range(1,22):
				if j>0 and j<22 and k>0 and k<22:
					#main body
					pixels_gr = [images_gr[i][j-1][k-1], images_gr[i][j-1][k], images_gr[i][j-1][k+1], images_gr[i][j][k-1], images_gr[i][j][k+1], images_gr[i][j+1][k-1], images_gr[i][j+1][k], images_gr[i][j+1][k+1]]  
					pixels_gi = [images_gi[i][j-1][k-1], images_gi[i][j-1][k], images_gi[i][j-1][k+1], images_gi[i][j][k-1], images_gi[i][j][k+1], images_gi[i][j+1][k-1], images_gi[i][j+1][k], images_gi[i][j+1][k+1]]
					pixels_gz = [images_gz[i][j-1][k-1], images_gz[i][j-1][k], images_gz[i][j-1][k+1], images_gz[i][j][k-1], images_gz[i][j][k+1], images_gz[i][j+1][k-1], images_gz[i][j+1][k], images_gz[i][j+1][k+1]]
					pixels_ri = [images_ri[i][j-1][k-1], images_ri[i][j-1][k], images_ri[i][j-1][k+1], images_ri[i][j][k-1], images_ri[i][j][k+1], images_ri[i][j+1][k-1], images_ri[i][j+1][k], images_ri[i][j+1][k+1]]
					pixels_rz = [images_rz[i][j-1][k-1], images_rz[i][j-1][k], images_rz[i][j-1][k+1], images_rz[i][j][k-1], images_rz[i][j][k+1], images_rz[i][j+1][k-1], images_rz[i][j+1][k], images_rz[i][j+1][k+1]]
					pixels_iz = [images_iz[i][j-1][k-1], images_iz[i][j-1][k], images_iz[i][j-1][k+1], images_iz[i][j][k-1], images_iz[i][j][k+1], images_iz[i][j+1][k-1], images_iz[i][j+1][k], images_iz[i][j+1][k+1]]


				if (np.max(images_gr[i][j][k]) > -99):
					if (abs(images_gr[i][j][k]) - abs(np.median(pixels_gr))) > 0.3:
						images_gr[i][j][k] = np.median(pixels_gr)
				if (np.max(images_gi[i][j][k]) > -99):
					if (abs(images_gi[i][j][k]) - abs(np.median(pixels_gi))) > 0.3:
						images_gi[i][j][k] = np.median(pixels_gi)
				if (np.max(images_gz[i][j][k]) > -99):
					if (abs(images_gz[i][j][k]) - abs(np.median(pixels_gz))) > 0.3:
						images_gz[i][j][k] = np.median(pixels_gz)
				if (np.max(images_ri[i][j][k]) > -99):
					if (abs(images_ri[i][j][k]) - abs(np.median(pixels_ri))) > 0.3:
						images_ri[i][j][k] = np.median(pixels_ri)
				if (np.max(images_rz[i][j][k]) > -99):
					if (abs(images_rz[i][j][k]) - abs(np.median(pixels_rz))) > 0.3:
						images_rz[i][j][k] = np.median(pixels_rz)
				if (np.max(images_iz[i][j][k]) > -99):
					if (abs(images_iz[i][j][k]) - abs(np.median(pixels_iz))) > 0.3:
						images_iz[i][j][k] = np.median(pixels_iz)


	for i in tqdm(range(0, len(images_g))):
		for j in range(0,23):
			for k in range(0,23):
			#corner and rim
				if j<1: #upper corners
					if k<1: #top left
						pixels_edge_gr = images_gr[i][j][k+1], images_gr[i][j+1][k], images_gr[i][j+1][k+1]
						pixels_edge_gi = images_gi[i][j][k+1], images_gi[i][j+1][k], images_gi[i][j+1][k+1]
						pixels_edge_gz = images_gz[i][j][k+1], images_gz[i][j+1][k], images_gz[i][j+1][k+1]
						pixels_edge_ri = images_ri[i][j][k+1], images_ri[i][j+1][k], images_ri[i][j+1][k+1]
						pixels_edge_rz = images_rz[i][j][k+1], images_rz[i][j+1][k], images_rz[i][j+1][k+1]
						pixels_edge_iz = images_iz[i][j][k+1], images_iz[i][j+1][k], images_iz[i][j+1][k+1]

					if k>21: #top right
						pixels_edge_gr = images_gr[i][j][k-1], images_gr[i][j+1][k], images_gr[i][j+1][k-1]
						pixels_edge_gi = images_gi[i][j][k-1], images_gi[i][j+1][k], images_gi[i][j+1][k-1]
						pixels_edge_gz = images_gz[i][j][k-1], images_gz[i][j+1][k], images_gz[i][j+1][k-1]
						pixels_edge_ri = images_ri[i][j][k-1], images_ri[i][j+1][k], images_ri[i][j+1][k-1]
						pixels_edge_rz = images_rz[i][j][k-1], images_rz[i][j+1][k], images_rz[i][j+1][k-1]
						pixels_edge_iz = images_iz[i][j][k-1], images_iz[i][j+1][k], images_iz[i][j+1][k-1]

					if k>0 and k<22: #top side
						pixels_edge_gr = images_gr[i][j][k-1], images_gr[i][j+1][k-1], images_gr[i][j+1][k], images_gr[i][j+1][k+1], images_gr[i][j][k+1]
						pixels_edge_gi = images_gi[i][j][k-1], images_gi[i][j+1][k-1], images_gi[i][j+1][k], images_gi[i][j+1][k+1], images_gi[i][j][k+1]
						pixels_edge_gz = images_gz[i][j][k-1], images_gz[i][j+1][k-1], images_gz[i][j+1][k], images_gz[i][j+1][k+1], images_gz[i][j][k+1]
						pixels_edge_ri = images_ri[i][j][k-1], images_ri[i][j+1][k-1], images_ri[i][j+1][k], images_ri[i][j+1][k+1], images_ri[i][j][k+1]
						pixels_edge_rz = images_rz[i][j][k-1], images_rz[i][j+1][k-1], images_rz[i][j+1][k], images_rz[i][j+1][k+1], images_rz[i][j][k+1]
						pixels_edge_iz = images_iz[i][j][k-1], images_iz[i][j+1][k-1], images_iz[i][j+1][k], images_iz[i][j+1][k+1], images_iz[i][j][k+1]

					if (np.max(images_gr[i][j][k]) > -99):
						if (abs(images_gr[i][j][k]) -abs(np.mean(pixels_edge_gr))) > 0.3:
							images_gr[i][j][k] = np.mean(pixels_edge_gr)
					if (np.max(images_gi[i][j][k]) > -99):
						if (abs(images_gi[i][j][k]) -abs(np.mean(pixels_edge_gi))) > 0.3:
							images_gi[i][j][k] = np.mean(pixels_edge_gi)
					if (np.max(images_gz[i][j][k]) > -99):
						if (abs(images_gz[i][j][k]) -abs(np.mean(pixels_edge_gz))) > 0.3:
							images_gz[i][j][k] = np.mean(pixels_edge_gz)
					if (np.max(images_ri[i][j][k]) > -99):
						if (abs(images_ri[i][j][k]) -abs(np.mean(pixels_edge_ri))) > 0.3:
							images_ri[i][j][k] = np.mean(pixels_edge_ri)
					if (np.max(images_rz[i][j][k]) > -99):
						if (abs(images_rz[i][j][k]) -abs(np.mean(pixels_edge_rz))) > 0.3:
							images_rz[i][j][k] = np.mean(pixels_edge_rz)
					if (np.max(images_iz[i][j][k]) > -99):
						if (abs(images_iz[i][j][k]) -abs(np.mean(pixels_edge_iz))) > 0.3:
							images_iz[i][j][k] = np.mean(pixels_edge_iz)







	for i in tqdm(range(0, len(images_g))):
		for j in range(0,23):
			for k in range(0,23):
				#left side right side
				if j>0 and j<22 and (k<1 or k>21):
					if k<1: #left side

						pixels_edge_gr = images_gr[i][j-1][k], images_gr[i][j-1][k+1], images_gr[i][j][k+1], images_gr[i][j+1][k+1], images_gr[i][j+1][k]
						pixels_edge_gi = images_gi[i][j-1][k], images_gi[i][j-1][k+1], images_gi[i][j][k+1], images_gi[i][j+1][k+1], images_gi[i][j+1][k]
						pixels_edge_gz = images_gz[i][j-1][k], images_gz[i][j-1][k+1], images_gz[i][j][k+1], images_gz[i][j+1][k+1], images_gz[i][j+1][k]
						pixels_edge_ri = images_ri[i][j-1][k], images_ri[i][j-1][k+1], images_ri[i][j][k+1], images_ri[i][j+1][k+1], images_ri[i][j+1][k]
						pixels_edge_rz = images_rz[i][j-1][k], images_rz[i][j-1][k+1], images_rz[i][j][k+1], images_rz[i][j+1][k+1], images_rz[i][j+1][k]
						pixels_edge_iz = images_iz[i][j-1][k], images_iz[i][j-1][k+1], images_iz[i][j][k+1], images_iz[i][j+1][k+1], images_iz[i][j+1][k]
					if k>21: #right side

						pixels_edge_gr = images_gr[i][j-1][k], images_gr[i][j-1][k-1], images_gr[i][j][k-1], images_gr[i][j+1][k-1], images_gr[i][j+1][k]
						pixels_edge_gi = images_gi[i][j-1][k], images_gi[i][j-1][k-1], images_gi[i][j][k-1], images_gi[i][j+1][k-1], images_gi[i][j+1][k]
						pixels_edge_gz = images_gz[i][j-1][k], images_gz[i][j-1][k-1], images_gz[i][j][k-1], images_gz[i][j+1][k-1], images_gz[i][j+1][k]
						pixels_edge_ri = images_ri[i][j-1][k], images_ri[i][j-1][k-1], images_ri[i][j][k-1], images_ri[i][j+1][k-1], images_ri[i][j+1][k]
						pixels_edge_rz = images_rz[i][j-1][k], images_rz[i][j-1][k-1], images_rz[i][j][k-1], images_rz[i][j+1][k-1], images_rz[i][j+1][k]
						pixels_edge_iz = images_iz[i][j-1][k], images_iz[i][j-1][k-1], images_iz[i][j][k-1], images_iz[i][j+1][k-1], images_iz[i][j+1][k]


					if (np.max(images_gr[i][j][k]) > -99):
						if (abs(images_gr[i][j][k]) -abs(np.mean(pixels_edge_gr))) > 0.3:
							images_gr[i][j][k] = np.mean(pixels_edge_gr)
					if (np.max(images_gi[i][j][k]) > -99):
						if (abs(images_gi[i][j][k]) -abs(np.mean(pixels_edge_gi))) > 0.3:
							images_gi[i][j][k] = np.mean(pixels_edge_gi)
					if (np.max(images_gz[i][j][k]) > -99):
						if (abs(images_gz[i][j][k]) -abs(np.mean(pixels_edge_gz))) > 0.3:
							images_gz[i][j][k] = np.mean(pixels_edge_gz)
					if (np.max(images_ri[i][j][k]) > -99):
						if (abs(images_ri[i][j][k]) -abs(np.mean(pixels_edge_ri))) > 0.3:
							images_ri[i][j][k] = np.mean(pixels_edge_ri)
					if (np.max(images_rz[i][j][k]) > -99):
						if (abs(images_rz[i][j][k]) -abs(np.mean(pixels_edge_rz))) > 0.3:
							images_rz[i][j][k] = np.mean(pixels_edge_rz)
					if (np.max(images_iz[i][j][k]) > -99):
						if (abs(images_iz[i][j][k]) -abs(np.mean(pixels_edge_iz))) > 0.3:
							images_iz[i][j][k] = np.mean(pixels_edge_iz)



	for i in tqdm(range(0, len(images_g))):
		for j in range(0,23):
			for k in range(0,23):
				if j>21: #lower corners
					if k<1: #lower left

						pixels_edge_gr = images_gr[i][j][k+1], images_gr[i][j-1][k], images_gr[i][j-1][k+1]
						pixels_edge_gi = images_gi[i][j][k+1], images_gi[i][j-1][k], images_gi[i][j-1][k+1]
						pixels_edge_gz = images_gz[i][j][k+1], images_gz[i][j-1][k], images_gz[i][j-1][k+1]
						pixels_edge_ri = images_ri[i][j][k+1], images_ri[i][j-1][k], images_ri[i][j-1][k+1]
						pixels_edge_rz = images_rz[i][j][k+1], images_rz[i][j-1][k], images_rz[i][j-1][k+1]
						pixels_edge_iz = images_iz[i][j][k+1], images_iz[i][j-1][k], images_iz[i][j-1][k+1]
					if k>21: #lower right

						pixels_edge_gr = images_gr[i][j][k-1], images_gr[i][j-1][k], images_gr[i][j-1][k-1]
						pixels_edge_gi = images_gi[i][j][k-1], images_gi[i][j-1][k], images_gi[i][j-1][k-1]
						pixels_edge_gz = images_gz[i][j][k-1], images_gz[i][j-1][k], images_gz[i][j-1][k-1]
						pixels_edge_ri = images_ri[i][j][k-1], images_ri[i][j-1][k], images_ri[i][j-1][k-1]
						pixels_edge_rz = images_rz[i][j][k-1], images_rz[i][j-1][k], images_rz[i][j-1][k-1]
						pixels_edge_iz = images_iz[i][j][k-1], images_iz[i][j-1][k], images_iz[i][j-1][k-1]

					if k>0 and k<22: #lower side

						pixels_edge_gr = images_gr[i][j][k-1], images_gr[i][j-1][k-1], images_gr[i][j-1][k], images_gr[i][j-1][k+1], images_gr[i][j][k+1]
						pixels_edge_gi = images_gi[i][j][k-1], images_gi[i][j-1][k-1], images_gi[i][j-1][k], images_gi[i][j-1][k+1], images_gi[i][j][k+1]
						pixels_edge_gz = images_gz[i][j][k-1], images_gz[i][j-1][k-1], images_gz[i][j-1][k], images_gz[i][j-1][k+1], images_gz[i][j][k+1]
						pixels_edge_ri = images_ri[i][j][k-1], images_ri[i][j-1][k-1], images_ri[i][j-1][k], images_ri[i][j-1][k+1], images_ri[i][j][k+1]
						pixels_edge_rz = images_rz[i][j][k-1], images_rz[i][j-1][k-1], images_rz[i][j-1][k], images_rz[i][j-1][k+1], images_rz[i][j][k+1]
						pixels_edge_iz = images_iz[i][j][k-1], images_iz[i][j-1][k-1], images_iz[i][j-1][k], images_iz[i][j-1][k+1], images_iz[i][j][k+1]

					if (np.max(images_gr[i][j][k]) > -99):
						if (abs(images_gr[i][j][k]) -abs(np.mean(pixels_edge_gr))) > 0.3:
							images_gr[i][j][k] = np.mean(pixels_edge_gr)
					if (np.max(images_gi[i][j][k]) > -99):
						if (abs(images_gi[i][j][k]) -abs(np.mean(pixels_edge_gi))) > 0.3:
							images_gi[i][j][k] = np.mean(pixels_edge_gi)
					if (np.max(images_gz[i][j][k]) > -99):
						if (abs(images_gz[i][j][k]) -abs(np.mean(pixels_edge_gz))) > 0.3:
							images_gz[i][j][k] = np.mean(pixels_edge_gz)
					if (np.max(images_ri[i][j][k]) > -99):
						if (abs(images_ri[i][j][k]) -abs(np.mean(pixels_edge_ri))) > 0.3:
							images_ri[i][j][k] = np.mean(pixels_edge_ri)
					if (np.max(images_rz[i][j][k]) > -99):
						if (abs(images_rz[i][j][k]) -abs(np.mean(pixels_edge_rz))) > 0.3:
							images_rz[i][j][k] = np.mean(pixels_edge_rz)
					if (np.max(images_iz[i][j][k]) > -99):
						if (abs(images_iz[i][j][k]) -abs(np.mean(pixels_edge_iz))) > 0.3:
							images_iz[i][j][k] = np.mean(pixels_edge_iz)




	clean_griz_cols = np.stack((images_gr, images_gi, images_gz, images_ri, images_rz, images_iz), axis=0)
	np.save(path + 'processed_dered_colours.npy', clean_griz_cols)


from astropy.table import Table
import pickle
import numpy as np
import sys

#load the observations
'''
griz  = np.load('/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/random_cats/dered_griz_obs.npy')
path = '/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/random_cats/'
i_g =griz[0] #[:100]
i_r =griz[1] #[:100]
i_i =griz[2] #[:100]
i_z =griz[3] #[:100]

#Fix noise pixels

images_g= []
images_r= []
images_i= []
images_z= []

pixel_lim = 0

for i in range(len(i_g)):
        images_g.append(pixel_threshold(i_g[i], pixel_lim))
        images_r.append(pixel_threshold(i_r[i], pixel_lim))
        images_i.append(pixel_threshold(i_i[i], pixel_lim))
        images_z.append(pixel_threshold(i_z[i], pixel_lim))

images_g= np.array(images_g)
images_r= np.array(images_r)
images_i= np.array(images_i)
images_z= np.array(images_z)


#clean images
clean_obs(images_g, images_r, images_i, images_z, path)
clean_cols(images_g, images_r, images_i, images_z, path)

'''

def area_map(radii):

	# Define the diameter of each pixel cube in arcseconds
	pixel_size_arcsec = 0.262


	# Initialize a grid to accumulate pixel values for all radii
	grid_size_all = int(2 * max(radii) / pixel_size_arcsec)
	total_grid = np.zeros((grid_size_all, grid_size_all))

	# Create 23x23 arrays
	array_all = np.zeros((grid_size_all, grid_size_all))
	all_areas = []

	for radius_arcsec in radii:
		# Calculate the number of pixels needed to fill the circular aperture
		area_aperture_arcsec2 = np.pi * (radius_arcsec**2)
		num_pixels = int(np.ceil(area_aperture_arcsec2 / pixel_size_arcsec**2))
		grid_size = int(np.ceil(np.sqrt(num_pixels)))
		grid = np.zeros((grid_size, grid_size))
		all_areas.append(area_aperture_arcsec2)

	    # Fill the pixels within the circular aperture
		for i in range(grid_size):
			for j in range(grid_size):
				x = (i + 0.5 - grid_size / 2) * pixel_size_arcsec
				y = (j + 0.5 - grid_size / 2) * pixel_size_arcsec
				if x**2 + y**2 <= (radius_arcsec**2):
					grid[i, j] = 1  # Inside the circular aperture

		# Determine the position to place the array within the grid
		position_x = (int(array_all.shape[0]) - grid_size) // 2
		position_y = (int(array_all.shape[0]) - grid_size) // 2

		# Copy the array into the grid
		array_all[position_x:position_x + grid_size, position_y:position_y + grid_size] += grid

	total_grid = total_grid + array_all

	# Slice the innermost 23x23 pixels
	inner_23x23 = total_grid[(grid_size_all - 23) // 2 : (grid_size_all + 23) // 2, (grid_size_all - 23) // 2 : (grid_size_all + 23) // 2]

	return all_areas, inner_23x23


########################################################


def ap_im_LS10(dat, all_areas, inner_23x23):

	bands = ['g', 'r', 'i', 'z']
	ap_images_LS10 = {'g': [], 'r': [], 'i': [], 'z': []}

	# Iterate over each band
	for u in range(len(bands)):
		band = bands[u]
		new = np.zeros((23, 23))
		for i in tqdm(range(len(dat))):  # Change this to the number of rows you want to process
			# Calculate the contributions of each value to the new array
			new = (
			(1 / all_areas[0]) * dat['dered_apflux_' + band + '_1'].iloc[i] * (inner_23x23 == 8) +
			(1 / (all_areas[1] - all_areas[0])) * (dat['dered_apflux_' + band + '_2'].iloc[i] - dat['dered_apflux_' + band + '_1'].iloc[i]) * (inner_23x23 == 7) +
			(1 / (all_areas[2] - all_areas[1])) * (dat['dered_apflux_' + band + '_3'].iloc[i] - dat['dered_apflux_' + band + '_2'].iloc[i]) * (inner_23x23 == 6) +
			(1 / (all_areas[3] - all_areas[2])) * (dat['dered_apflux_' + band + '_4'].iloc[i] - dat['dered_apflux_' + band + '_3'].iloc[i]) * (inner_23x23 == 5) +
			(1 / (all_areas[4] - all_areas[3])) * (dat['dered_apflux_' + band + '_5'].iloc[i] - dat['dered_apflux_' + band + '_4'].iloc[i]) * (inner_23x23 == 4) +
			(1 / (all_areas[5] - all_areas[4])) * (dat['dered_apflux_' + band + '_6'].iloc[i] - dat['dered_apflux_' + band + '_5'].iloc[i]) * (inner_23x23 == 3) +
			(1 / (all_areas[6] - all_areas[5])) * (dat['dered_apflux_' + band + '_7'].iloc[i] - dat['dered_apflux_' + band + '_6'].iloc[i]) * (inner_23x23 == 2)
			)

			# Append the result to the respective band list
			ap_images_LS10[band].append(new)

		ap_images_LS10[band] = np.array(ap_images_LS10[band])
		#print(f"{band}-band data shape: {ap_images_LS10[band].shape}")


	return ap_images_LS10


########################################################


def ap_im_LS10_ivar(dat, all_areas, inner_23x23, ap_im_LS10):

	bands = ['g', 'r', 'i', 'z']
	ap_images_LS10_ivar = {'g': [], 'r': [], 'i': [], 'z': []}
	max_vals = []
	min_vals = []

	# Iterate over each band
	for u in range(len(bands)):
		band = bands[u]
		new = np.zeros((23, 23))
		for i in tqdm(range(len(dat))):  # Change this to the number of rows you want to process
			# Calculate the contributions of each value to the new array
			new = (
			dat['apflux_ivar_' + band + '_1'].iloc[i] * (inner_23x23 == 8) +
			dat['apflux_ivar_' + band + '_2'].iloc[i] * (inner_23x23 == 7) +
			dat['apflux_ivar_' + band + '_3'].iloc[i] * (inner_23x23 == 6) +
			dat['apflux_ivar_' + band + '_4'].iloc[i] * (inner_23x23 == 5) +
			dat['apflux_ivar_' + band + '_5'].iloc[i] * (inner_23x23 == 4) +
			dat['apflux_ivar_' + band + '_6'].iloc[i] * (inner_23x23 == 3) +
			dat['apflux_ivar_' + band + '_7'].iloc[i] * (inner_23x23 == 2)
			)

			# Append the result to the respective band list
			ap_images_LS10_ivar[band].append(new)

		ap_images_LS10_ivar[band] = np.array(ap_im_LS10[band] * np.sqrt(ap_images_LS10_ivar[band]))
		#ap_images_LS10_ivar[band] = np.array(ap_images_LS10_ivar[band])
		#print(f"{band}-band data shape: {ap_images_LS10_ivar[band].shape}")
		max_vals.append(np.max(ap_images_LS10_ivar[band]))
		min_vals.append(np.min(ap_images_LS10_ivar[band]))


	for j in range(len(bands)):
		band = bands[j]
		ap_images_LS10_ivar[band] = (ap_images_LS10_ivar[band] - np.min(min_vals)) / (np.max(max_vals) - np.min(min_vals))

	return ap_images_LS10_ivar


########################################################


def ap_im_WISE(dat, all_areas_wise, inner_23x23):

        bands = ['w1', 'w2', 'w3', 'w4']
        ap_images_WISE = {'w1': [], 'w2': [], 'w3': [], 'w4': []}

        # Iterate over each band
        for u in range(len(bands)):
                band = bands[u]
                new = np.zeros((23, 23))
                for i in tqdm(range(len(dat))):  # Change this to the number of rows you want to process
                        # Calculate the contributions of each value to the new array
                        new = (
			(1 / all_areas_wise[0]) * dat['dered_apflux_' + band + '_1'].iloc[i] * (inner_23x23 == 2) +
			(1 / (all_areas_wise[1] - all_areas_wise[0])) * (dat['dered_apflux_' + band + '_2'].iloc[i] - dat['dered_apflux_' + band + '_1'].iloc[i]) * (inner_23x23 == 1)
                        )

                        # Append the result to the respective band list
                        ap_images_WISE[band].append(new)

                ap_images_WISE[band] = np.array(ap_images_WISE[band])
                #print(f"{band}-band data shape: {ap_images_WISE[band].shape}")


        return ap_images_WISE


#######################################################


def ap_im_WISE_ivar(dat, all_areas_wise, inner_23x23, ap_im_WISE):

	bands = ['w1', 'w2', 'w3', 'w4']
	ap_images_WISE_ivar = {'w1': [], 'w2': [], 'w3': [], 'w4': []}
	max_vals = []
	min_vals = []

	# Iterate over each band
	for u in range(len(bands)):
		band = bands[u]
		new = np.zeros((23, 23))
		for i in tqdm(range(len(dat))):  # Change this to the number of rows you want to process
                        # Calculate the contributions of each value to the new array
			new = (
			dat['apflux_ivar_' + band + '_1'].iloc[i] * (inner_23x23 == 2) +
			dat['apflux_ivar_' + band + '_2'].iloc[i] * (inner_23x23 == 1)
			)

			# Append the result to the respective band list
			ap_images_WISE_ivar[band].append(new)

		ap_images_WISE_ivar[band] = np.array(ap_im_WISE[band] * np.sqrt(ap_images_WISE_ivar[band]))
		#ap_images_WISE_ivar[band] = np.array(ap_images_WISE_ivar[band])
		#print(f"{band}-band data shape: {ap_images_WISE_ivar[band].shape}")
		max_vals.append(np.max(ap_images_WISE_ivar[band]))
		min_vals.append(np.min(ap_images_WISE_ivar[band]))


	for j in range(len(bands)):
		band = bands[j]
		ap_images_WISE_ivar[band] = (ap_images_WISE_ivar[band] - np.min(min_vals)) / (np.max(max_vals) - np.min(min_vals))


	return ap_images_WISE_ivar


#######################################################


def ap_im_WISE_res(dat, all_areas_wise, inner_23x23):

        bands = ['w1', 'w2', 'w3', 'w4']
        ap_images_WISE_res = {'w1': [], 'w2': [], 'w3': [], 'w4': []}

        # Iterate over each band
        for u in range(len(bands)):
                band = bands[u]
                new = np.zeros((23, 23))
                for i in tqdm(range(len(dat))):  # Change this to the number of rows you want to process
                        # Calculate the contributions of each value to the new array
                        new = (
                        dat['apflux_resid_' + band + '_1'].iloc[i] * (inner_23x23 == 2) +
                        dat['apflux_resid_' + band + '_2'].iloc[i] * (inner_23x23 == 1)
                        )

                        # Append the result to the respective band list
                        ap_images_WISE_res[band].append(new)

                ap_images_WISE_res[band] = np.array(ap_images_WISE_res[band])
                #print(f"{band}-band data shape: {ap_images_WISE_res[band].shape}")


        return ap_images_WISE_res


#######################################################


def ap_cols(ap_images_LS10,ap_images_WISE):

	ap_images_g_LS10 = ap_images_LS10['g']
	ap_images_r_LS10 = ap_images_LS10['r']
	ap_images_i_LS10 = ap_images_LS10['i']
	ap_images_z_LS10 = ap_images_LS10['z']

	ap_images_w1_WISE = ap_images_WISE['w1']
	ap_images_w2_WISE = ap_images_WISE['w2']
	ap_images_w3_WISE = ap_images_WISE['w3']
	ap_images_w4_WISE = ap_images_WISE['w4']

#######

	ap_images_gr_LS10=np.zeros((len(ap_images_g_LS10),23,23))
	ap_images_gi_LS10=np.zeros((len(ap_images_g_LS10),23,23))
	ap_images_gz_LS10=np.zeros((len(ap_images_g_LS10),23,23))
	ap_images_ri_LS10=np.zeros((len(ap_images_g_LS10),23,23))
	ap_images_rz_LS10=np.zeros((len(ap_images_g_LS10),23,23))
	ap_images_iz_LS10=np.zeros((len(ap_images_g_LS10),23,23))

	ap_images_w12_WISE=np.zeros((len(ap_images_g_LS10),23,23))
	ap_images_w13_WISE=np.zeros((len(ap_images_g_LS10),23,23))
	ap_images_w14_WISE=np.zeros((len(ap_images_g_LS10),23,23))
	ap_images_w23_WISE=np.zeros((len(ap_images_g_LS10),23,23))
	ap_images_w24_WISE=np.zeros((len(ap_images_g_LS10),23,23))
	ap_images_w34_WISE=np.zeros((len(ap_images_g_LS10),23,23))

	default = -99


	for i in range(len(ap_images_g_LS10)):

		if (np.min(ap_images_g_LS10[i]) >0) & (np.min(ap_images_r_LS10[i]) >0):
			ap_images_gr_LS10[i] = (22.5-2.5*np.log10(ap_images_g_LS10[i])) - (22.5-2.5*np.log10(ap_images_r_LS10[i]))
		else:
			ap_images_gr_LS10[i] = default

		if (np.min(ap_images_g_LS10[i]) >0) & (np.min(ap_images_i_LS10[i]) >0):
			ap_images_gi_LS10[i] = (22.5-2.5*np.log10(ap_images_g_LS10[i])) - (22.5-2.5*np.log10(ap_images_i_LS10[i]))
		else:
			ap_images_gi_LS10[i] = default

		if (np.min(ap_images_g_LS10[i]) >0) & (np.min(ap_images_z_LS10[i]) >0):
			ap_images_gz_LS10[i] = (22.5-2.5*np.log10(ap_images_g_LS10[i])) - (22.5-2.5*np.log10(ap_images_z_LS10[i]))
		else:
			ap_images_gz_LS10[i] = default

		if (np.min(ap_images_r_LS10[i]) >0) & (np.min(ap_images_i_LS10[i]) >0):
			ap_images_ri_LS10[i] = (22.5-2.5*np.log10(ap_images_r_LS10[i])) - (22.5-2.5*np.log10(ap_images_i_LS10[i]))
		else:
			ap_images_ri_LS10[i] = default

		if (np.min(ap_images_r_LS10[i]) >0) & (np.min(ap_images_z_LS10[i]) >0):
			ap_images_r_LS10[i] = (22.5-2.5*np.log10(ap_images_r_LS10[i])) - (22.5-2.5*np.log10(ap_images_z_LS10[i]))
		else:
			ap_images_rz_LS10[i] = default

		if (np.min(ap_images_i_LS10[i]) >0) & (np.min(ap_images_z_LS10[i]) >0):
			ap_images_iz_LS10[i] = (22.5-2.5*np.log10(ap_images_i_LS10[i])) - (22.5-2.5*np.log10(ap_images_z_LS10[i]))
		else:
			ap_images_iz_LS10[i] = default

###


		if (np.min(ap_images_w1_WISE[i]) >0) & (np.min(ap_images_w2_WISE[i]) >0):
			ap_images_w12_WISE[i] = (22.5-2.5*np.log10(ap_images_w1_WISE[i])) - (22.5-2.5*np.log10(ap_images_w2_WISE[i]))
		else:
			ap_images_w12_WISE[i] = default

		if (np.min(ap_images_w1_WISE[i]) >0) & (np.min(ap_images_w3_WISE[i]) >0):
			ap_images_w13_WISE[i] = (22.5-2.5*np.log10(ap_images_w1_WISE[i])) - (22.5-2.5*np.log10(ap_images_w3_WISE[i]))
		else:
			ap_images_w13_WISE[i] = default

		if (np.min(ap_images_w1_WISE[i]) >0) & (np.min(ap_images_w4_WISE[i]) >0):
			ap_images_w14_WISE[i] = (22.5-2.5*np.log10(ap_images_w1_WISE[i])) - (22.5-2.5*np.log10(ap_images_w4_WISE[i]))
		else:
			ap_images_w14_WISE[i] = default

		if (np.min(ap_images_w2_WISE[i]) >0) & (np.min(ap_images_w3_WISE[i]) >0):
			ap_images_w23_WISE[i] = (22.5-2.5*np.log10(ap_images_w2_WISE[i])) - (22.5-2.5*np.log10(ap_images_w3_WISE[i]))
		else:
			ap_images_w23_WISE[i] = default

		if (np.min(ap_images_w2_WISE[i]) >0) & (np.min(ap_images_w4_WISE[i]) >0):
			ap_images_w24_WISE[i] = (22.5-2.5*np.log10(ap_images_w2_WISE[i])) - (22.5-2.5*np.log10(ap_images_w4_WISE[i]))
		else:
			ap_images_w24_WISE[i] = default

		if (np.min(ap_images_w3_WISE[i]) >0) & (np.min(ap_images_w4_WISE[i]) >0):
			ap_images_w34_WISE[i] = (22.5-2.5*np.log10(ap_images_w3_WISE[i])) - (22.5-2.5*np.log10(ap_images_w4_WISE[i]))
		else:
			ap_images_w34_WISE[i] = default



	ap_ims_LS10_cols = np.stack((ap_images_gr_LS10,ap_images_gi_LS10,ap_images_gz_LS10,ap_images_ri_LS10,ap_images_rz_LS10, ap_images_iz_LS10), axis=0)
	ap_ims_WISE_cols = np.stack((ap_images_w12_WISE,ap_images_w13_WISE,ap_images_w14_WISE,ap_images_w23_WISE, ap_images_w24_WISE,ap_images_w34_WISE), axis=0)


	return ap_ims_LS10_cols, ap_ims_WISE_cols




file_path = "/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/random_cats/"
data = Table.read(file_path + 'LS_n2.fits', hdu=1)
dataset = data.to_pandas()

dataset_ext = cat_preprocessing.run_all_preprocessing(dataset)

# Define the radii of the circular apertures in arcseconds
radii_LS10 = [0.5 , 0.75, 1.0, 1.5 , 2.0 , 3.5, 5.0, 7.0]
radii_WISE = [3.0, 5.0]

#LS10 ap im + ivar
area_LS10, inner_23x23 = area_map(radii_LS10)
ap_im_LS10 = ap_im_LS10(dataset_ext, area_LS10, inner_23x23)
ap_im_LS10_ivar = ap_im_LS10_ivar(dataset_ext, area_LS10, inner_23x23, ap_im_LS10)

#WISE ap im + ivar + res
area_WISE, inner_23x23 = area_map(radii_WISE)
ap_im_WISE = ap_im_WISE(dataset_ext, area_WISE, inner_23x23)
ap_im_WISE_ivar = ap_im_WISE_ivar(dataset_ext, area_WISE, inner_23x23, ap_im_WISE)
ap_im_WISE_res = ap_im_WISE_res(dataset_ext, area_WISE, inner_23x23)

#LS10 & WISE ap colours
ap_ims_LS10_cols, ap_ims_WISE_cols = ap_cols(ap_im_LS10,ap_im_WISE)

#print(ap_ims_LS10_cols.shape)
suffix = '_LRG'

np.save(file_path + 'aperture_images_LS10' + suffix + '.npy', ap_im_LS10, allow_pickle=True)
np.save(file_path + 'aperture_images_LS10_ivar' + suffix + '.npy', ap_im_LS10_ivar, allow_pickle=True)
np.save(file_path + 'aperture_images_WISE' + suffix + '.npy', ap_im_WISE, allow_pickle=True)
np.save(file_path + 'aperture_images_WISE_ivar' + suffix + '.npy', ap_im_WISE_ivar, allow_pickle=True)
np.save(file_path + 'aperture_images_WISE_residuals' + suffix + '.npy', ap_im_WISE_res, allow_pickle=True)
np.save(file_path + 'aperture_images_LS10_colours' + suffix + '.npy', ap_ims_LS10_cols, allow_pickle=True)
np.save(file_path + 'aperture_images_WISE_colours' + suffix + '.npy', ap_ims_WISE_cols, allow_pickle=True)
