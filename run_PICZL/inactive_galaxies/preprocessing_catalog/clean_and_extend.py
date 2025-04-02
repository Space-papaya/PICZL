

'''

#####  CLEAN AND EXTEND  #####

This file and the functions therein will be called first to preprare the input catalogue (features).

The catalog on which these functions can be run requires following characteristics:

        - can be either a csv or fits file format, however must be converted to pandas dataframe prior to running this script
        - needs to have all 310 columns retrieved from the Legacy Survey
        - ...

'''


#####################################
#Importing libraries and dependancies
#####################################


import numpy as np
from numpy import load
import pandas as pd
from tqdm import tqdm
from math import log
import sys
import warnings

# Suppress PerformanceWarnings from pandas
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------



def run_all_preprocessing(dataset):

	'''
	Input: dataset
	Output: dataset

	This function calls all the functions defined above and returns the preprocessed dataset, the original dataset, noise and colour features names for LS10 and WISE.
	'''

	#Running all functions defined above
	print(' >> Original catalog:')
	dataset = type_one_hot_encoding(dataset)
	dataset = fix_corrupted_fluxes(dataset)
	dataset = dereden_fluxes_add_colour_features(dataset)
	dataset = define_additional_catalogue_features(dataset)
	print('\n >> Processed catalog:')
	print(dataset)

	return dataset



################################################



# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------



def type_one_hot_encoding(dataset):

	'''
	Input: dataset
	Output: dataset + (original dataset)

	This function one-hot encodes the 5 dimensional coloumn "type" and return a new dataset. If desired, the dataset can be saved in its (original) form for later comparison.
	'''

	#Add training sample (TS) ID
	dataset['type'] = dataset['type'].str.decode('utf-8')
	dataset['ORIG_TYPE'] = dataset['ORIG_TYPE'].str.decode('utf-8')
	dataset['FULLID'] = dataset['FULLID'].str.decode('utf-8')
	dataset['Cat'] = dataset['Cat'].str.decode('utf-8')
	print(dataset)
	dataset['TS_ID'] = dataset.index


	#One hot encode the coloumn 'type' and adds the generated coloumns to dataset.
	#Caution: "type" refers to the definition made by the authors, i.e. the type as defined by the largest DCHISQ value.
	#The type defined by the Legacy Survey can be found under "ORIG_TYPE".
	one_hot = pd.get_dummies(dataset['type'],dtype=int)
	dataset = dataset.join(one_hot)


	return pd.DataFrame(dataset) #316 coloumns



################################################



# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------




def fix_corrupted_fluxes(dataset):
	'''
	Input: dataset
	Output: dataset

	This function cleans the numerical deredened total flux and aperture flux entries of the dataset. For this, all NaN, negative infinity or positive infinity are replaced by default.
	'''

	#Fix LS10 fluxes (g,r,i,z) and WISE fluxes (w1,w2,w3,w4) to be > 0
	fluxes = ['g','r','i','z','w1','w2','w3','w4']
	count=0

	#Set default to overwrite non-physical flux measurements
	default = 0

	for i in fluxes:
		#Fill NaN values with random noise
		dataset['dered_flux_'+i].fillna(default, inplace=True)
		#Replace +/- infinity values with random noise
		dataset['dered_flux_'+i].replace([-np.inf, np.inf], default, inplace=True)
		#Replace non-positive values with random noise
		dataset['dered_flux_'+i] = dataset['dered_flux_'+i].apply(lambda x: default if x < 0 else x)

		if count < 4:
			#LS10 fluxes have 8 apertures
			for l in range(1,9):
				dataset['apflux_'+i+'_'+str(l)].fillna(default, inplace=True)
				dataset['apflux_'+i+'_'+str(l)].replace([-np.inf, np.inf], default, inplace=True)
				dataset['apflux_'+i+'_'+str(l)] = dataset['apflux_'+i+'_'+str(l)].apply(lambda x: default if x < 0 else x)

		else:
			#WISE fluxes have 5 apertures
			for l in range(1,6):
				dataset['apflux_'+i+'_'+str(l)].fillna(default, inplace=True)
				dataset['apflux_'+i+'_'+str(l)].replace([-np.inf, np.inf], default, inplace=True)
				dataset['apflux_'+i+'_'+str(l)] = dataset['apflux_'+i+'_'+str(l)].apply(lambda x: default if x < 0 else x)

		#Increment count variable
		count=count+1

	return dataset



# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------



def dereden_fluxes_add_colour_features(dataset):
	'''
	Input: dataset
	Output: dataset, aperture_colour feature names legacy and aperture_colour feature names wise

	This function deredens the observed aperture fluxes, by dividing each band by its respective milky way transmission along the line of sight, depending on the objects location on sky.
	Additionally all colours for deredened total and aperture fluxes are calculated.
	'''

	#Dereden fluxes by considering the milky way absorption, "red" prefix in features name referring to non-deredned fluxes
	red_apflux_WISE = ["apflux_w1_1", "apflux_w1_2", "apflux_w1_3", "apflux_w1_4", "apflux_w1_5", "apflux_w2_1", "apflux_w2_2", "apflux_w2_3", "apflux_w2_4", "apflux_w2_5", \
			 "apflux_w3_1", "apflux_w3_2", "apflux_w3_3", "apflux_w3_4", "apflux_w3_5", "apflux_w4_1", "apflux_w4_2", "apflux_w4_3", "apflux_w4_4", "apflux_w4_5"]
	red_apflux_LS10 = ["apflux_g_1", "apflux_g_2", "apflux_g_3", "apflux_g_4", "apflux_g_5", "apflux_g_6", "apflux_g_7", "apflux_g_8", "apflux_r_1", "apflux_r_2", "apflux_r_3", \
			"apflux_r_4", "apflux_r_5", "apflux_r_6", "apflux_r_7", "apflux_r_8", "apflux_i_1", "apflux_i_2", "apflux_i_3", "apflux_i_4", "apflux_i_5", "apflux_i_6", "apflux_i_7", \
			"apflux_i_8", "apflux_z_1", "apflux_z_2", "apflux_z_3", "apflux_z_4", "apflux_z_5", "apflux_z_6", "apflux_z_7", "apflux_z_8"]
	transmission_WISE = ["mw_transmission_w1","mw_transmission_w2","mw_transmission_w3","mw_transmission_w4"]
	transmission_LS10 = ["mw_transmission_g","mw_transmission_r","mw_transmission_i","mw_transmission_z"]

	#Increment variable & deredstacking list
	k=0
	new_dered_apflux_WISE = []
	new_dered_apflux_LS10 = []

	#Loop over the aperture fluxes and calculate the dereddened values
	#WISE fluxes have 5 apertures
	for i in range(0, len(red_apflux_WISE)):
		new_dered_ap_w = pd.DataFrame(dataset.apply(lambda row: row[red_apflux_WISE[i]] / row[transmission_WISE[k]] if row[red_apflux_WISE[k]] > 0 else 0, axis=1))
		new_dered_ap_w.columns = ["dered_" + red_apflux_WISE[i]]
		new_dered_apflux_WISE.append(new_dered_ap_w)
		if ((i + 1) % 5) == 0:
			k += 1

	k = 0

	#LS10 fluxes have 8 apertures
	for i in range(0, len(red_apflux_LS10)):
		new_dered_ap_l = pd.DataFrame(dataset.apply(lambda row: row[red_apflux_LS10[i]] / row[transmission_LS10[k]] if row[red_apflux_LS10[k]] > 0 else 0, axis=1))
		new_dered_ap_l.columns = ["dered_" + red_apflux_LS10[i]]
		new_dered_apflux_LS10.append(new_dered_ap_l)
		if ((i + 1) % 8) == 0:
			k += 1

	#Combine all dereddened fluxes
	new_dered_apflux_WISE = pd.concat(new_dered_apflux_WISE, axis=1)
	new_dered_apflux_LS10 = pd.concat(new_dered_apflux_LS10, axis=1)
	all_dered_apflux = new_dered_apflux_WISE.join(new_dered_apflux_LS10)

	#Remove aperture fluxes (and transmissions) from catalogue since they have been replaced with deredened fluxes
	dataset = dataset.join(all_dered_apflux)
	dataset = dataset.drop(red_apflux_WISE,axis = 1)
	dataset = dataset.drop(red_apflux_LS10,axis = 1)
	dataset = dataset.drop(transmission_WISE,axis = 1)
	dataset = dataset.drop(transmission_LS10,axis = 1)


	# -------------------------------------------------------------------------------------
	# Compute colours for deredened total fluxes
	# -------------------------------------------------------------------------------------


	#Add 28 colours (from total dered_flux). These fluxes are > 0 by default.
	cols = ['g_r', 'g_i', 'g_z', 'r_i', 'r_z', 'i_z','g_w1', 'g_w2', 'g_w3', 'g_w4', 'r_w1', 'r_w2', 'r_w3', 'r_w4', 'i_w1', 'i_w2', 'i_w3', 'i_w4', \
		'z_w1', 'z_w2', 'z_w3', 'z_w4', 'w1_w2', 'w1_w3', 'w1_w4', 'w2_w3', 'w2_w4', 'w3_w4']

	#Create a list of DataFrames for the new columns
	new_cols = []

	#We compute the colours as 22.5 - (2.5 * log10)
	for col in cols:
		if col.startswith(('g_', 'r_', 'i_', 'z_')):
			mask = (dataset['dered_flux_' + col[0]] > 0) & (dataset['dered_flux_' + col[2:]] >0)
			# Create a new DataFrame with -99 as the default value
			new_col = pd.DataFrame(-99, index=dataset.index, columns=[col])
			new_col.loc[mask] = (22.5 - (2.5*np.log10(dataset.loc[mask]['dered_flux_' + col[0]]))) - (22.5 - (2.5*np.log10(dataset.loc[mask]['dered_flux_' + col[2:]])))

		elif col.startswith(('w1_', 'w2_', 'w3_', 'w4_')):
			mask = (dataset['dered_flux_' + col[:2]] > 0) & (dataset['dered_flux_' + col[3:]] >0)
			# Create a new DataFrame with -99 as the default value
			new_col = pd.DataFrame(-99, index=dataset.index, columns=[col])
			new_col.loc[mask] = (22.5 - (2.5*np.log10(dataset.loc[mask]['dered_flux_' + col[:2]]))) - (22.5 - (2.5*np.log10(dataset.loc[mask]['dered_flux_' + col[3:]])))
		new_col.columns = [col]
		new_cols.append(new_col)
	#Concatenate all the new columns
	new_cols_combined = pd.concat(new_cols, axis=1)

	#Add the new columns to the dataset
	dataset = pd.concat([dataset, new_cols_combined], axis=1)



	# -------------------------------------------------------------------------------------
	# Compute colours for deredened aperture fluxes
	# -------------------------------------------------------------------------------------



	#Add 78 colours (from dered_aperture, 30 WISE, 48 LS10). Again all ap_fluxes are > 0 by default.

	c=5  #WISE apertures

	feature_names_WISE =[]
	feature_names_LS10 = []
	new_ap_cols = []

	#Computing aperture colour for adjacent apertures, w1 to w2, w2 to w3, w3 to w4 (for all 5 apertures)
	for i in range(0,15):
		feature_name = str(red_apflux_WISE[i])[-4:-2]+str(red_apflux_WISE[i+c])[-3:-2]+str(red_apflux_WISE[i])[-2:-1]+"ap"+str(red_apflux_WISE[i])[-1:]
		mask = (dataset["dered_" + red_apflux_WISE[i]] > 0) & (dataset["dered_" + red_apflux_WISE[i + c]] > 0)
		# Create a default DataFrame with -99 values
		new_ap_col = pd.DataFrame(-99, index=dataset.index, columns=[feature_name])
		new_ap_col.loc[mask] = (22.5 - (2.5 * np.log10(dataset.loc[mask]["dered_" + red_apflux_WISE[i]]))) - (22.5 - (2.5 * np.log10(dataset.loc[mask]\
					["dered_" + red_apflux_WISE[i + c]])))

		new_ap_col.name = feature_name
		new_ap_cols.append(new_ap_col)
		feature_names_WISE.append(feature_name)

		#Computing aperture colour for single-skip adjacent apertures, w1 to w3, w2 to w4 (for all 5 apertures)
		if i < 10:
			feature_name = str(red_apflux_WISE[i])[-4:-2]+str(red_apflux_WISE[i+(c*2)])[-3:-2]+str(red_apflux_WISE[i])[-2:-1]+"ap"+str(red_apflux_WISE[i])[-1:]
			mask = (dataset["dered_" + red_apflux_WISE[i]] > 0) & (dataset["dered_" + red_apflux_WISE[i + (c*2)]] > 0)
			# Create a default DataFrame with -99 values
			new_ap_col = pd.DataFrame(-99, index=dataset.index, columns=[feature_name])
			new_ap_col.loc[mask] = (22.5 - (2.5 * np.log10(dataset.loc[mask]["dered_" + red_apflux_WISE[i]]))) - (22.5 - (2.5 * np.log10(dataset.loc[mask]\
						["dered_" + red_apflux_WISE[i + (c*2)]])))

			new_ap_col.name = feature_name
			new_ap_cols.append(new_ap_col)
			feature_names_WISE.append(feature_name)


		#Computing aperture colour for double-skip adjacent apertures, w1 to w4 (for all 5 apertures)
		if i < 5:
			feature_name = str(red_apflux_WISE[i])[-4:-2]+str(red_apflux_WISE[i+(c*3)])[-3:-2]+str(red_apflux_WISE[i])[-2:-1]+"ap"+str(red_apflux_WISE[i])[-1:]
			mask = (dataset["dered_" + red_apflux_WISE[i]] > 0) & (dataset["dered_" + red_apflux_WISE[i + (c*3)]] > 0)
			# Create a default DataFrame with -99 values
			new_ap_col = pd.DataFrame(-99, index=dataset.index, columns=[feature_name])
			new_ap_col.loc[mask] = (22.5 - (2.5 * np.log10(dataset.loc[mask]["dered_" + red_apflux_WISE[i]]))) - (22.5 - (2.5 * np.log10(dataset.loc[mask]\
						["dered_" + red_apflux_WISE[i + (c*3)]])))

			new_ap_col.name = feature_name
			new_ap_cols.append(new_ap_col)
			feature_names_WISE.append(feature_name)




	c=8  #LS10 apertures

	#Computing aperture colour for adjacent apertures, see WISE description
	for i in range(0,24):
		feature_name = str(red_apflux_LS10[i])[-3:-2]+str(red_apflux_LS10[i+c])[-3:-2]+str(red_apflux_LS10[i])[-2:-1]+"ap"+str(red_apflux_LS10[i])[-1:]
		mask = (dataset["dered_" + red_apflux_LS10[i]] > 0) & (dataset["dered_" + red_apflux_LS10[i + c]] > 0)
		# Create a default DataFrame with -99 values
		new_ap_col = pd.DataFrame(-99, index=dataset.index, columns=[feature_name])
		new_ap_col.loc[mask] = (22.5 - (2.5 * np.log10(dataset.loc[mask]["dered_" + red_apflux_LS10[i]]))) - (22.5 - (2.5 * np.log10(dataset.loc[mask]\
					["dered_" + red_apflux_LS10[i + c]])))

		new_ap_col.name = feature_name
		new_ap_cols.append(new_ap_col)
		feature_names_LS10.append(feature_name)

		#Computing aperture colour for single-skip adjacent apertures
		if i < 16:
			feature_name = str(red_apflux_LS10[i])[-3:-2]+str(red_apflux_LS10[i+(c*2)])[-3:-2]+str(red_apflux_LS10[i])[-2:-1]+"ap"+str(red_apflux_LS10[i])[-1:]
			mask = (dataset["dered_" + red_apflux_LS10[i]] > 0) & (dataset["dered_" + red_apflux_LS10[i + (c*2)]] > 0)
			# Create a default DataFrame with -99 values
			new_ap_col = pd.DataFrame(-99, index=dataset.index, columns=[feature_name])
			new_ap_col.loc[mask] = (22.5 - (2.5 * np.log10(dataset.loc[mask]["dered_" + red_apflux_LS10[i]]))) - (22.5 - (2.5 * np.log10(dataset.loc[mask]\
						["dered_" + red_apflux_LS10[i + (c*2)]])))

			new_ap_col.name = feature_name
			new_ap_cols.append(new_ap_col)
			feature_names_LS10.append(feature_name)

		#Computing aperture colour for double-skip adjacent apertures
		if i < 8:
			feature_name = str(red_apflux_LS10[i])[-3:-2]+str(red_apflux_LS10[i+(c*3)])[-3:-2]+str(red_apflux_LS10[i])[-2:-1]+"ap"+str(red_apflux_LS10[i])[-1:]
			mask = (dataset["dered_" + red_apflux_LS10[i]] > 0) & (dataset["dered_" + red_apflux_LS10[i + (c*3)]] > 0)
			# Create a default DataFrame with -99 values
			new_ap_col = pd.DataFrame(-99, index=dataset.index, columns=[feature_name])
			new_ap_col.loc[mask] = (22.5 - (2.5 * np.log10(dataset.loc[mask]["dered_" + red_apflux_LS10[i]]))) - (22.5 - (2.5 * np.log10(dataset.loc[mask]\
						["dered_" + red_apflux_LS10[i + (c*3)]])))

			new_ap_col.name = feature_name
			new_ap_cols.append(new_ap_col)
			feature_names_LS10.append(feature_name)


	#Concatenate all the new columns at once
	new_ap_cols_combined = pd.concat(new_ap_cols, axis=1)

	#Add the new columns to the dataset
	dataset = pd.concat([dataset, new_ap_cols_combined], axis=1)

	return dataset




# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------



def define_additional_catalogue_features(dataset):
	'''
	Input: dataset
	Output: dataset

	This function cleans catalog features which potentially could introduce nans and adds further features, such as maskbits, to the dataset.
	'''

	#Cleaning original features
	ivar_features = ['flux_ivar_g', 'flux_ivar_r', 'flux_ivar_i', 'flux_ivar_z', 'flux_ivar_w1', 'flux_ivar_w2', 'flux_ivar_w3', 'flux_ivar_w4']
	dataset[ivar_features].fillna(0)
	dataset[ivar_features].replace([-np.inf, np.inf], 0)

	frac_features = ['fracflux_g', 'fracflux_r', 'fracflux_i', 'fracflux_z', 'fracflux_w1', 'fracflux_w2', 'fracflux_w3', 'fracflux_w4']
	dataset[frac_features].fillna(0)
	dataset[frac_features].replace([-np.inf, np.inf], 0)



	return dataset























