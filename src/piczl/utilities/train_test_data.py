
import numpy as np
from numpy import load
import pandas as pd
from sklearn.model_selection import train_test_split


def arrange_tt_features(images, images_col, combined_non_2D_features, index, labels):
    """
    Splits the input data into training and testing sets.

    This function takes as input:
        - 2D flux images,
        - 2D color images,
        - 1D catalog features,
        - source indices,
        - and corresponding redshift labels.

    It returns train/test splits of all inputs using a fixed random seed for reproducibility.

    Parameters:
        images (np.ndarray): Array of shape (N, H, W, C) with 2D flux images.
        images_col (np.ndarray): Array of shape (N, H, W, C) with 2D color images.
        combined_non_2D_features (np.ndarray): Array of shape (N, D) with catalog features.
        index (np.ndarray): Array of shape (N,) with object/source indices.
        labels (np.ndarray): Array of shape (N, 1) with true redshift labels.

    Returns:
        Tuple:
            train_images (np.ndarray): Training set of flux images
            test_images (np.ndarray): Test set of flux images
            train_labels (np.ndarray): Training redshift labels
            test_labels (np.ndarray): Test redshift labels
            train_features (np.ndarray): Training catalog features
            test_features (np.ndarray): Test catalog features
            train_ind (np.ndarray): Training indices
            test_ind (np.ndarray): Test indices
            train_col_images (np.ndarray): Training set of color images
            test_col_images (np.ndarray): Test set of color images
    """
    random_state = 42


    train_images, test_images, train_labels, test_labels, train_features, test_features, train_ind, test_ind = train_test_split(
        images, labels, combined_non_2D_features, index, test_size=0.2, random_state=random_state
    )


    train_col_images, test_col_images = train_test_split(
        images_col, test_size=0.2, random_state=random_state
    )

    print("Train flux images shape: " + str(train_images.shape))
    print("Train colour images shape: " + str(train_col_images.shape))
    print("Train catalog features shape: " + str(train_features.shape))
    print("Train labels shape: " + str(train_labels.shape))

    return (
        train_images,
        test_images,
        train_labels,
        test_labels,
        train_features,
        test_features,
        train_ind,
        test_ind,
        train_col_images,
        test_col_images,
    )
