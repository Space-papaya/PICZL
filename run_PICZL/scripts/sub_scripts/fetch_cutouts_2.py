


#Import libraries

import pandas as pd
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import shutil
import glob
from astropy.table import Table
import time
import urllib.request
import io
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import concurrent.futures
from urllib.error import HTTPError
import sys


##########################################################
##########################################################


def get_data(df, type):

	RA = np.array(df['RA'])
	DEC = np.array(df['DEC'])
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

