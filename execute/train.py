import os
import tensorflow as tf
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from piczl.core.trainer import run_trainer
from piczl.utilities import *

DATA_PATH = '/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/FLASH_comp/'

with tf.device('/GPU:0'):
	run_trainer(
	catalog_path=DATA_PATH + "combined_FLASH_PICZL_ready.fits",
	image_path=DATA_PATH,
	mode = "new",
	sub_sample = False,
	max_sources=20
	)

