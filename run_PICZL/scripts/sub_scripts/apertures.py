

import numpy as np
from tqdm import tqdm


########################################################
########################################################


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


#######################################################




