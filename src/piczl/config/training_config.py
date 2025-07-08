import tensorflow as tf
import pickle
import numpy as np
import sys
from tensorflow.keras import backend as K
import gc
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from piczl.utilities import *


def run_models(
    loss_func,
    epochs,
    batch_sizes,
    num_gaussian,
    learning_rates,
    version,
    features,
    train_images,
    train_col_images,
    train_features,
    train_labels,
    test_images,
    test_col_images,
    test_features,
    test_labels,
):
    """
    Train multiple models over a grid of hyperparameters and save results.

    Parameters:
        loss_func (function): Loss function to use (e.g. CRPS or NLL).
        epochs (int): Number of training epochs.
        batch_sizes (list): List of batch sizes to try.
        num_gaussian (list): List of numbers of GMM components to use.
        learning_rates (list): List of learning rates to try.
        version (str): Version tag for saving checkpoints.
        features (np.ndarray): Input tabular features.
        train_images (np.ndarray): Training image data.
        train_col_images (np.ndarray): Training color image data.
        train_features (np.ndarray): Training tabular features.
        train_labels (np.ndarray): Training redshift labels.
        test_images (np.ndarray): Testing image data.
        test_col_images (np.ndarray): Testing color image data.
        test_features (np.ndarray): Testing tabular features.
        test_labels (np.ndarray): Testing redshift labels.

    Returns:
        None. Saves predictions and training history to disk.
    """

    lf = "CRPS" if loss_func == loss_functions.crps_loss else "NLL"

    all_histories_and_configs = []
    all_predictions = []
    all_train_predictions = []

    model_counter = 0
    for num_gauss in num_gaussian:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                model_counter += 1
                tf.keras.backend.clear_session()  # Clear session between models

                model = get_model.compile_model(
                    features.shape[1], num_gauss, learning_rate, loss_func
                )

                history, model, checkpoint_dir = train_model.train_model(
                    model,
                    epochs,
                    batch_size,
                    learning_rate,
                    loss_func,
                    version,
                    train_images,
                    train_col_images,
                    train_features,
                    train_labels,
                    test_images,
                    test_col_images,
                    test_features,
                    test_labels,
                )

                print(
                    f"Model {model_counter} trained. "
                    f"Validation Loss: {min(history.history['val_loss'])}"
                )

                preds = model.predict([test_images, test_col_images, test_features])
                all_predictions.append(preds)

                config = {
                    "gmm_components": num_gauss,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                }
                history_and_config = {
                    "config": config,
                    "history": history.history,
                }
                all_histories_and_configs.append(history_and_config)

                del model
                K.clear_session()
                gc.collect()

    with open(os.path.join(checkpoint_dir, "hist.pkl"), "wb") as pickle_file:
        pickle.dump(all_histories_and_configs, pickle_file)

    with open(os.path.join(checkpoint_dir, "preds.pkl"), "wb") as file:
        pickle.dump(all_predictions, file)


