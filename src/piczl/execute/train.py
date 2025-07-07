import os
import tensorflow as tf
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from piczl.core.trainer import run_trainer
from piczl.utilities import *

# Checking available GPU's, optional: set memory limit
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        gpu_configuration.limit_gpu_memory(memory_gb=25)
        print("GPU enabled:", gpus[0])
    except RuntimeError as e:
        print("Failed to set GPU configuration:", e)
else:
    print("No GPU available. Running on CPU.")



def train_new_models(DATA_PATH=None, catalog_name=None, subsample=True, use_demo_data=True):


    if use_demo_data:
        catalog_path, image_path = load_demo_data.get_demo_data_path('train')
    else:
        catalog_path = os.path.join(DATA_PATH, catalog_name)
        image_data = DATA_PATH

    gpus = tf.config.list_physical_devices('GPU')
    device = '/GPU:0' if gpus else 'CPU'
    print(f"Training on device: {device}")

    with tf.device(device):
        run_trainer(
            catalog_path=catalog_path,
            image_path=image_path,
            mode='new',
            sub_sample=subsample,
            max_sources=20
        )


