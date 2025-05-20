
from astropy.table import Table
import pickle
import numpy as np
import sys
from sub_scripts.fetch_cutouts import *


file_path = "/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/FLASH_comp/"
data = Table.read(file_path + 'combined_FLASH_PICZL_ready.fits', hdu=1)
dataset = data.to_pandas()

dataset_sub = dataset.iloc[:].reset_index(drop=True)
j=20000
i=55
#for i in range(0,100):

dataset_s = dataset_sub.iloc[j+(i*100):j+((i+1)*100)].reset_index(drop=True)

#Get the observations
dered_griz_obs = get_data(dataset_s, "obs")
np.save(file_path + "obs/dered_griz_obs_"+str(j)+"_"+str(i)+".npy", dered_griz_obs)

