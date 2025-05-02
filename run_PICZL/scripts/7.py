
from astropy.table import Table
import pickle
import numpy as np
import sys
from sub_scripts.fetch_cutouts import *


file_path = "/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/Ching/"
data = Table.read(file_path + 'Ching_PICZL_ready.fits', hdu=1)
dataset = data.to_pandas()
#dataset_sub = dataset.iloc[2000:5000].reset_index(drop=True)

for i in range(34,200):
	dataset_sub = dataset.iloc[4000+(i*5):4000+((i+1)*5)].reset_index(drop=True)

	#Get the observations
	dered_griz_obs = get_data(dataset_sub, "resid")
	np.save(file_path +"resid/" + "dered_griz_resid_4000_"+str(i)+".npy", dered_griz_obs)


##Get the observations
#dered_griz_resid = get_data(dataset_sub, "resid")
#np.save(file_path + "dered_griz_resid_2.npy", dered_griz_resid)
