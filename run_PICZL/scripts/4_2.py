
from astropy.table import Table
import pickle
import numpy as np
import sys
from sub_scripts.fetch_cutouts import *


file_path = "/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/MRC_1J/"
data = Table.read(file_path + 'MRC_1J_PICZL_ready.fits', hdu=1)
dataset = data.to_pandas()
dataset_sub = dataset.iloc[:].reset_index(drop=True)


#Get the observations
dered_griz_model = get_data(dataset_sub, "model")
np.save(file_path + "dered_griz_model.npy", dered_griz_model)
