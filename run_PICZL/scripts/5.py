
from astropy.table import Table
import pickle
import numpy as np
import sys
from sub_scripts.clean import *
from sub_scripts.apertures import *


file_path = "/home/wroster/learning-photoz/PICZL_OZ/train_PICZL/data/"
data = Table.read(file_path + 'TS_3pt.fits', hdu=1)
datase = data.to_pandas()

dataset = datase.iloc[80447:].reset_index(drop=True)

#file_path = "/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/FLASH_comp/"
#data = Table.read(file_path + 'combined_FLASH_PICZL_ready.fits', hdu=1)
#dataset = data.to_pandas()
#dataset_sub = dataset.head(10).reset_index(drop=True)

dataset_ext = run_all_preprocessing(dataset)

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

np.save(file_path + 'ap_im_LS10' + suffix + '.npy', ap_im_LS10, allow_pickle=True)
np.save(file_path + 'ap_im_LS10_ivar' + suffix + '.npy', ap_im_LS10_ivar, allow_pickle=True)
np.save(file_path + 'ap_im_WISE' + suffix + '.npy', ap_im_WISE, allow_pickle=True)
np.save(file_path + 'ap_im_WISE_ivar' + suffix + '.npy', ap_im_WISE_ivar, allow_pickle=True)
np.save(file_path + 'ap_im_WISE_res' + suffix + '.npy', ap_im_WISE_res, allow_pickle=True)
np.save(file_path + 'ap_ims_LS10_cols' + suffix + '.npy', ap_ims_LS10_cols, allow_pickle=True)
np.save(file_path + 'ap_ims_WISE_cols' + suffix + '.npy', ap_ims_WISE_cols, allow_pickle=True)
