
from astropy.table import Table
import pickle
import numpy as np
import sys
from sub_scripts.fetch_cutouts_2 import *


file_path = "/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/FLASH_comp/"
data = Table.read(file_path + 'combined_FLASH_PICZL_ready.fits', hdu=1)
dataset = data.to_pandas()
#dataset_sub = dataset.head(8000).reset_index(drop=True)


#Get the observations
dered_griz_obs = get_data(dataset, "model")
np.save(file_path + "dered_griz_model.npy", dered_griz_obs)
#dered_griz_model = get_data(dataset, "model")
#np.save(file_path + "dered_griz_model.npy", dered_griz_model)
#dered_griz_resid = get_data(dataset, "resid")
#np.save(file_path + "dered_griz_resid.npy", dered_griz_resid)
#print(dered_griz_obs.shape)
