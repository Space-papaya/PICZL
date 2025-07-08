import os
import tensorflow as tf
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from piczl.core.trainer import run_trainer
from piczl.utilities import *


def train_new_models(
    DATA_PATH=None, catalog_name=None, subsample=True, use_demo_data=True
):
    """
    Trains a new model using either demo data or user-provided data.

    Parameters:
        DATA_PATH (str or None): The root path to the user-provided data. Ignored if use_demo_data is True.
        catalog_name (str or None): The name of the catalog FITS file (if not using demo data).
        subsample (bool): Whether to randomly subsample the training data (useful for testing/debugging).
        use_demo_data (bool): If True, loads demo catalog and image data bundled in the package.

    Returns:
        None
    """

    if use_demo_data:
        catalog_path, image_path = load_demo_data.get_demo_data_path("train")
    else:
        catalog_path = os.path.join(DATA_PATH, catalog_name)
        image_path = DATA_PATH

    device = gpu_configuration.set_computing()
    with tf.device(device):
        run_trainer(
            catalog_path=catalog_path,
            image_path=image_path,
            mode="new",
            sub_sample=subsample,
            max_sources=20,
        )
