import os
import tensorflow as tf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from piczl.core.estimator import run_estimation

DATA_PATH = '/home/wroster/learning-photoz/PICZL_OZ/examples/'

with tf.device('/GPU:0'):
    run_estimation(
        catalog_path=DATA_PATH + "catalog.fits",
        image_path=DATA_PATH + "images/",
        mode='inactive',
        max_sources=20
    )
