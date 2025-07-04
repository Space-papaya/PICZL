"""
preprocessing.py

This module contains functions for preprocessing the photometric dataset,
including fixing corrupted flux values, dereddening fluxes, adding color features,
and defining additional catalog features. The main entry point is
`run_all_preprocessing` which runs all steps and returns the processed dataset.

Dependencies:
- numpy
- pandas
- math
"""

import numpy as np
import pandas as pd
from math import log
import sys
import warnings

# Suppress PerformanceWarnings from pandas
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def run_all_preprocessing(dataset):
    """
    Runs the full preprocessing pipeline on the input dataset.

    This function sequentially applies:
    - One-hot encoding of galaxy types
    - Fixing corrupted flux values (NaN, inf, negative)
    - Dereddening fluxes and adding color features
    - Defining additional catalog features

    Parameters
    ----------
    dataset : pd.DataFrame
        Input photometric catalog as a pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        The fully preprocessed catalog with added features.
    """
    dataset = type_one_hot_encoding(dataset)
    dataset = fix_corrupted_fluxes(dataset)
    dataset = dereden_fluxes_add_colour_features(dataset)
    dataset = define_additional_catalogue_features(dataset)

    return pd.DataFrame(dataset)



def type_one_hot_encoding(dataset):
    """
    One-hot encodes the 'type' column in the dataset and adds relevant metadata.

    This function performs the following operations:
    - Decodes byte strings in columns 'type', 'ORIG_TYPE', 'FULLID', and 'Cat' to UTF-8 strings.
    - Adds a new column 'TS_ID' set to the dataset's index values (training sample IDs).
    - One-hot encodes the 'type' column (defined by the largest DCHISQ value).
    - Joins the one-hot encoded columns back to the original dataset.

    Note:
    - The 'type' column is author-defined.
    - The original Legacy Survey type is in the 'ORIG_TYPE' column.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataset containing columns 'type', 'ORIG_TYPE', 'FULLID', and 'Cat'.

    Returns
    -------
    pd.DataFrame
        Dataset with one-hot encoded 'type' columns added and TS_ID column.
    """

    dataset['type'] = dataset['type'].str.decode('utf-8')
    dataset['ORIG_TYPE'] = dataset['ORIG_TYPE'].str.decode('utf-8')
    dataset['FULLID'] = dataset['FULLID'].str.decode('utf-8')
    dataset['Cat'] = dataset['Cat'].str.decode('utf-8')

    dataset['TS_ID'] = dataset.index
    one_hot = pd.get_dummies(dataset['type'], dtype=int)
    dataset = dataset.join(one_hot)

    return pd.DataFrame(dataset)



def fix_corrupted_fluxes(dataset):
    """
    Clean corrupted numerical flux entries in the dataset.

    This function processes dereddened total flux and aperture flux measurements
    from LS10 (g, r, i, z bands) and WISE (w1, w2, w3, w4 bands). It replaces
    all NaN, positive infinity, and negative infinity values with a default value.
    Additionally, any negative flux values are replaced by the default.

    Aperture fluxes for LS10 bands have 8 apertures, and WISE bands have 5 apertures.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataset containing dereddened flux columns and aperture flux columns.

    Returns
    -------
    pd.DataFrame
        Dataset with corrupted flux values fixed/replaced by default values.
    """
    fluxes = ['g', 'r', 'i', 'z', 'w1', 'w2', 'w3', 'w4']
    default = 0
    count = 0

    for band in fluxes:
        dataset[f'dered_flux_{band}'].fillna(default, inplace=True)
        dataset[f'dered_flux_{band}'].replace([-np.inf, np.inf], default, inplace=True)
        dataset[f'dered_flux_{band}'] = dataset[f'dered_flux_{band}'].apply(lambda x: default if x < 0 else x)

        if count < 4:
            for aperture in range(1, 9):
                col = f'apflux_{band}_{aperture}'
                dataset[col].fillna(default, inplace=True)
                dataset[col].replace([-np.inf, np.inf], default, inplace=True)
                dataset[col] = dataset[col].apply(lambda x: default if x < 0 else x)
        else:
            for aperture in range(1, 6):
                col = f'apflux_{band}_{aperture}'
                dataset[col].fillna(default, inplace=True)
                dataset[col].replace([-np.inf, np.inf], default, inplace=True)
                dataset[col] = dataset[col].apply(lambda x: default if x < 0 else x)

        count += 1

    return pd.DataFrame(dataset)



def dereden_fluxes_add_colour_features(dataset):
    '''
    Deredden aperture fluxes and compute color features from both total and aperture fluxes.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Input data frame containing observed aperture fluxes and Milky Way transmission values 
        for WISE and LS10 bands, as well as total fluxes.

    Returns
    -------
    dataset : pandas.DataFrame
        The input dataset augmented with:
        - Dereddened aperture fluxes replacing the original aperture fluxes.
        - New color features computed from dereddened total fluxes.
        - New color features computed from dereddened aperture fluxes for WISE and LS10 bands.

    Additional Returns (implied in original docstring)
    -----------------------------------------------
    feature_names_WISE : list of str
        Names of the aperture color features derived from WISE bands.

    feature_names_LS10 : list of str
        Names of the aperture color features derived from LS10 bands.
    '''

    red_apflux_WISE = ["apflux_w1_1", "apflux_w1_2", "apflux_w1_3", "apflux_w1_4", "apflux_w1_5", "apflux_w2_1", "apflux_w2_2", "apflux_w2_3", "apflux_w2_4", "apflux_w2_5", \
                     "apflux_w3_1", "apflux_w3_2", "apflux_w3_3", "apflux_w3_4", "apflux_w3_5", "apflux_w4_1", "apflux_w4_2", "apflux_w4_3", "apflux_w4_4", "apflux_w4_5"]
    red_apflux_LS10 = ["apflux_g_1", "apflux_g_2", "apflux_g_3", "apflux_g_4", "apflux_g_5", "apflux_g_6", "apflux_g_7", "apflux_g_8", "apflux_r_1", "apflux_r_2", "apflux_r_3", \
                    "apflux_r_4", "apflux_r_5", "apflux_r_6", "apflux_r_7", "apflux_r_8", "apflux_i_1", "apflux_i_2", "apflux_i_3", "apflux_i_4", "apflux_i_5", "apflux_i_6", "apflux_i_7", \
                    "apflux_i_8", "apflux_z_1", "apflux_z_2", "apflux_z_3", "apflux_z_4", "apflux_z_5", "apflux_z_6", "apflux_z_7", "apflux_z_8"]
    transmission_WISE = ["mw_transmission_w1","mw_transmission_w2","mw_transmission_w3","mw_transmission_w4"]
    transmission_LS10 = ["mw_transmission_g","mw_transmission_r","mw_transmission_i","mw_transmission_z"]

    k=0
    new_dered_apflux_WISE = []
    new_dered_apflux_LS10 = []

    for i in range(0, len(red_apflux_WISE)):
        new_dered_ap_w = pd.DataFrame(dataset.apply(lambda row: row[red_apflux_WISE[i]] / row[transmission_WISE[k]] if row[red_apflux_WISE[k]] > 0 else 0, axis=1))
        new_dered_ap_w.columns = ["dered_" + red_apflux_WISE[i]]
        new_dered_apflux_WISE.append(new_dered_ap_w)
        if ((i + 1) % 5) == 0:
            k += 1

    k = 0

    for i in range(0, len(red_apflux_LS10)):
        new_dered_ap_l = pd.DataFrame(dataset.apply(lambda row: row[red_apflux_LS10[i]] / row[transmission_LS10[k]] if row[red_apflux_LS10[k]] > 0 else 0, axis=1))
        new_dered_ap_l.columns = ["dered_" + red_apflux_LS10[i]]
        new_dered_apflux_LS10.append(new_dered_ap_l)
        if ((i + 1) % 8) == 0:
            k += 1

    new_dered_apflux_WISE = pd.concat(new_dered_apflux_WISE, axis=1)
    new_dered_apflux_LS10 = pd.concat(new_dered_apflux_LS10, axis=1)
    all_dered_apflux = new_dered_apflux_WISE.join(new_dered_apflux_LS10)

    dataset = dataset.join(all_dered_apflux)
    dataset = dataset.drop(red_apflux_WISE,axis = 1)
    dataset = dataset.drop(red_apflux_LS10,axis = 1)
    dataset = dataset.drop(transmission_WISE,axis = 1)
    dataset = dataset.drop(transmission_LS10,axis = 1)

    cols = ['g_r', 'g_i', 'g_z', 'r_i', 'r_z', 'i_z','g_w1', 'g_w2', 'g_w3', 'g_w4', 'r_w1', 'r_w2', 'r_w3', 'r_w4', 'i_w1', 'i_w2', 'i_w3', 'i_w4', \
        'z_w1', 'z_w2', 'z_w3', 'z_w4', 'w1_w2', 'w1_w3', 'w1_w4', 'w2_w3', 'w2_w4', 'w3_w4']

    new_cols = []

    for col in cols:
        if col.startswith(('g_', 'r_', 'i_', 'z_')):
            mask = (dataset['dered_flux_' + col[0]] > 0) & (dataset['dered_flux_' + col[2:]] >0)
            new_col = pd.DataFrame(-99, index=dataset.index, columns=[col])
            new_col.loc[mask] = (22.5 - (2.5*np.log10(dataset.loc[mask]['dered_flux_' + col[0]]))) - (22.5 - (2.5*np.log10(dataset.loc[mask]['dered_flux_' + col[2:]])))
        elif col.startswith(('w1_', 'w2_', 'w3_', 'w4_')):
            mask = (dataset['dered_flux_' + col[:2]] > 0) & (dataset['dered_flux_' + col[3:]] >0)
            new_col = pd.DataFrame(-99, index=dataset.index, columns=[col])
            new_col.loc[mask] = (22.5 - (2.5*np.log10(dataset.loc[mask]['dered_flux_' + col[:2]]))) - (22.5 - (2.5*np.log10(dataset.loc[mask]['dered_flux_' + col[3:]])))
        new_col.columns = [col]
        new_cols.append(new_col)

    new_cols_combined = pd.concat(new_cols, axis=1)
    dataset = pd.concat([dataset, new_cols_combined], axis=1)

    c=5

    feature_names_WISE =[]
    feature_names_LS10 = []
    new_ap_cols = []

    for i in range(0,15):
        feature_name = str(red_apflux_WISE[i])[-4:-2]+str(red_apflux_WISE[i+c])[-3:-2]+str(red_apflux_WISE[i])[-2:-1]+"ap"+str(red_apflux_WISE[i])[-1:]
        mask = (dataset["dered_" + red_apflux_WISE[i]] > 0) & (dataset["dered_" + red_apflux_WISE[i + c]] > 0)
        new_ap_col = pd.DataFrame(-99, index=dataset.index, columns=[feature_name])
        new_ap_col.loc[mask] = (22.5 - (2.5 * np.log10(dataset.loc[mask]["dered_" + red_apflux_WISE[i]]))) - (22.5 - (2.5 * np.log10(dataset.loc[mask]\
                    ["dered_" + red_apflux_WISE[i + c]])))
        new_ap_col.name = feature_name
        new_ap_cols.append(new_ap_col)
        feature_names_WISE.append(feature_name)

        if i < 10:
            feature_name = str(red_apflux_WISE[i])[-4:-2]+str(red_apflux_WISE[i+(c*2)])[-3:-2]+str(red_apflux_WISE[i])[-2:-1]+"ap"+str(red_apflux_WISE[i])[-1:]
            mask = (dataset["dered_" + red_apflux_WISE[i]] > 0) & (dataset["dered_" + red_apflux_WISE[i + (c*2)]] > 0)
            new_ap_col = pd.DataFrame(-99, index=dataset.index, columns=[feature_name])
            new_ap_col.loc[mask] = (22.5 - (2.5 * np.log10(dataset.loc[mask]["dered_" + red_apflux_WISE[i]]))) - (22.5 - (2.5 * np.log10(dataset.loc[mask]\
                        ["dered_" + red_apflux_WISE[i + (c*2)]])))
            new_ap_col.name = feature_name
            new_ap_cols.append(new_ap_col)
            feature_names_WISE.append(feature_name)

        if i < 5:
            feature_name = str(red_apflux_WISE[i])[-4:-2]+str(red_apflux_WISE[i+(c*3)])[-3:-2]+str(red_apflux_WISE[i])[-2:-1]+"ap"+str(red_apflux_WISE[i])[-1:]
            mask = (dataset["dered_" + red_apflux_WISE[i]] > 0) & (dataset["dered_" + red_apflux_WISE[i + (c*3)]] > 0)
            new_ap_col = pd.DataFrame(-99, index=dataset.index, columns=[feature_name])
            new_ap_col.loc[mask] = (22.5 - (2.5 * np.log10(dataset.loc[mask]["dered_" + red_apflux_WISE[i]]))) - (22.5 - (2.5 * np.log10(dataset.loc[mask]\
                        ["dered_" + red_apflux_WISE[i + (c*3)]])))
            new_ap_col.name = feature_name
            new_ap_cols.append(new_ap_col)
            feature_names_WISE.append(feature_name)

    c=8

    for i in range(0,24):
        feature_name = str(red_apflux_LS10[i])[-3:-2]+str(red_apflux_LS10[i+c])[-3:-2]+str(red_apflux_LS10[i])[-2:-1]+"ap"+str(red_apflux_LS10[i])[-1:]
        mask = (dataset["dered_" + red_apflux_LS10[i]] > 0) & (dataset["dered_" + red_apflux_LS10[i + c]] > 0)
        new_ap_col = pd.DataFrame(-99, index=dataset.index, columns=[feature_name])
        new_ap_col.loc[mask] = (22.5 - (2.5 * np.log10(dataset.loc[mask]["dered_" + red_apflux_LS10[i]]))) - (22.5 - (2.5 * np.log10(dataset.loc[mask]\
                    ["dered_" + red_apflux_LS10[i + c]])))
        new_ap_col.name = feature_name
        new_ap_cols.append(new_ap_col)
        feature_names_LS10.append(feature_name)

        if i < 16:
            feature_name = str(red_apflux_LS10[i])[-3:-2]+str(red_apflux_LS10[i+(c*2)])[-3:-2]+str(red_apflux_LS10[i])[-2:-1]+"ap"+str(red_apflux_LS10[i])[-1:]
            mask = (dataset["dered_" + red_apflux_LS10[i]] > 0) & (dataset["dered_" + red_apflux_LS10[i + (c*2)]] > 0)
            new_ap_col = pd.DataFrame(-99, index=dataset.index, columns=[feature_name])
            new_ap_col.loc[mask] = (22.5 - (2.5 * np.log10(dataset.loc[mask]["dered_" + red_apflux_LS10[i]]))) - (22.5 - (2.5 * np.log10(dataset.loc[mask]\
                        ["dered_" + red_apflux_LS10[i + (c*2)]])))
            new_ap_col.name = feature_name
            new_ap_cols.append(new_ap_col)
            feature_names_LS10.append(feature_name)

        if i < 8:
            feature_name = str(red_apflux_LS10[i])[-3:-2]+str(red_apflux_LS10[i+(c*3)])[-3:-2]+str(red_apflux_LS10[i])[-2:-1]+"ap"+str(red_apflux_LS10[i])[-1:]
            mask = (dataset["dered_" + red_apflux_LS10[i]] > 0) & (dataset["dered_" + red_apflux_LS10[i + (c*3)]] > 0)
            new_ap_col = pd.DataFrame(-99, index=dataset.index, columns=[feature_name])
            new_ap_col.loc[mask] = (22.5 - (2.5 * np.log10(dataset.loc[mask]["dered_" + red_apflux_LS10[i]]))) - (22.5 - (2.5 * np.log10(dataset.loc[mask]\
                        ["dered_" + red_apflux_LS10[i + (c*3)]])))
            new_ap_col.name = feature_name
            new_ap_cols.append(new_ap_col)
            feature_names_LS10.append(feature_name)

    new_ap_cols_combined = pd.concat(new_ap_cols, axis=1)
    dataset = pd.concat([dataset, new_ap_cols_combined], axis=1)

    return pd.DataFrame(dataset)



def define_additional_catalogue_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and augment catalogue features that might contain NaNs or infinite values.

    This function targets inverse variance ('flux_ivar_*') and fraction flux ('fracflux_*')
    features in the dataset, replacing NaNs and infinities with zeros to ensure clean inputs.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Input catalogue containing photometric features to be cleaned.

    Returns
    -------
    pandas.DataFrame
        Dataset with cleaned flux inverse variance and fraction flux features.
    """
    ivar_features = [
        'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_i', 'flux_ivar_z',
        'flux_ivar_w1', 'flux_ivar_w2', 'flux_ivar_w3', 'flux_ivar_w4'
    ]
    dataset[ivar_features] = dataset[ivar_features].fillna(0)
    dataset[ivar_features] = dataset[ivar_features].replace([-np.inf, np.inf], 0)

    frac_features = [
        'fracflux_g', 'fracflux_r', 'fracflux_i', 'fracflux_z',
        'fracflux_w1', 'fracflux_w2', 'fracflux_w3', 'fracflux_w4'
    ]
    dataset[frac_features] = dataset[frac_features].fillna(0)
    dataset[frac_features] = dataset[frac_features].replace([-np.inf, np.inf], 0)

    return pd.DataFrame(dataset)
