import os
import tensorflow as tf
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from piczl.core.estimator import run_estimation
from piczl.utilities import *

DATA_PATH = '/home/wroster/learning-photoz/PICZL_OZ/tests/example_data/small/'

psf = False
sub_sample = False
max_sources = 20

catalog_path = DATA_PATH + 'example_cat.fits'

dataset, image_data = load_data.fetch_all_inputs(catalog_path, DATA_PATH, psf=psf, sub_sample_yesno=sub_sample, sub_sample_size=max_sources)


