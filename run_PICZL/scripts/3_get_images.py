
from astropy.table import Table
import pickle
import numpy as np
import sys
from sub_scripts.fetch_cutouts import *


file_path = "/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/Ching/"
data = Table.read(file_path + 'Ching_PICZL_ready.fits', hdu=1)
dataset = data.to_pandas()



for i in range(235,500):
	dataset_sub = dataset.iloc[8500+(i*5):8500+((i+1)*5)].reset_index(drop=True)

	#Get the observations
	dered_griz_obs = get_data(dataset_sub, "obs")
	np.save(file_path +"obs_2/" + "dered_griz_obs_8500_"+str(i)+".npy", dered_griz_obs)

