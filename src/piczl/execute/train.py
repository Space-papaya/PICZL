import os
import tensorflow as tf
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from piczl.core.trainer import run_trainer
from piczl.utilities import *


def train_new_models(
    DATA_PATH=None, catalog_name=None, subsample=True, use_demo_data=True
):
    if use_demo_data:
        catalog_path, image_path = load_demo_data.get_demo_data_path("train")
    else:
        catalog_path = os.path.join(DATA_PATH, catalog_name)
        image_data = DATA_PATH

    device = gpu_configuration.set_computing()
    with tf.device(device):
        run_trainer(
            catalog_path=catalog_path,
            image_path=image_path,
            mode="new",
            sub_sample=subsample,
            max_sources=20,
        )
