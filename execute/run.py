import os
import tensorflow as tf
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from piczl.core.estimator import run_estimation
from piczl.utilities import *

DATA_PATH = '/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/FLASH_comp/'

with tf.device('/GPU:0'):
	z_modes, l1s, u1s, degeneracy, dataset = run_estimation(
	catalog_path=DATA_PATH + "combined_FLASH_PICZL_ready.fits",
	image_path=DATA_PATH,
	mode='inactive',
	sub_sample = False,
	max_sources=20
	)

print("z_peak:", z_modes[:5])
print("l1s:", l1s[:5])
print("u1s:", u1s[:5])
print("degeneracy:", degeneracy[:5])

pwd = '/home/wroster/learning-photoz/PICZL_OZ/do_feats_matter/'
catalog_name = 'active_feat'
output.append_output(dataset, pwd, catalog_name, z_modes, l1s, u1s, degeneracy)
