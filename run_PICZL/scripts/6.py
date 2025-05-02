

from astropy.table import Table
import pickle
import numpy as np
import sys
from sub_scripts.colours import *

#load the observations
griz  = np.load('/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/Ching/dered_griz_obs.npy')
i_g =griz[0] #[:100]
i_r =griz[1] #[:100]
i_i =griz[2] #[:100]
i_z =griz[3] #[:100]

#Fix noise pixels

images_g= []
images_r= []
images_i= []
images_z= []

pixel_lim = 0

for i in range(len(i_g)):
        images_g.append(pixel_threshold(i_g[i], pixel_lim))
        images_r.append(pixel_threshold(i_r[i], pixel_lim))
        images_i.append(pixel_threshold(i_i[i], pixel_lim))
        images_z.append(pixel_threshold(i_z[i], pixel_lim))

images_g= np.array(images_g)
images_r= np.array(images_r)
images_i= np.array(images_i)
images_z= np.array(images_z)


#clean images
clean_obs(images_g, images_r, images_i, images_z)
clean_cols(images_g, images_r, images_i, images_z)

