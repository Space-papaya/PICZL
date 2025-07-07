"""
image_processing.py

Utility functions for handling and stacking image data arrays.
"""

import numpy as np


def stack_images(image_data):
    """
    Separates and stacks image data arrays from a dictionary based on their key names.

    This function extracts two groups of images from `image_data`:
    - Those whose keys end with 'col' (e.g., color features)
    - Those that do not (e.g., spatial or PSF images)

    It stacks each group of arrays along the last axis to produce two combined image arrays.

    Args:
        image_data (dict): Dictionary of numpy arrays representing image channels/features,
                           where keys are feature names.

    Returns:
        tuple:
            images (np.ndarray): Stacked array of image data whose keys do NOT end with 'col'.
            images_col (np.ndarray): Stacked array of image data whose keys end with 'col'.

    Example:
        images, images_col = stack_images(image_data)
    """
    # Separate keys based on suffix 'col'
    col_variables = [
        var_name for var_name in image_data.keys() if var_name.endswith("col")
    ]
    non_col_variables = [
        var_name for var_name in image_data.keys() if not var_name.endswith("col")
    ]

    # Extract arrays in the same order as keys
    col_arrays = [image_data[var] for var in col_variables]
    non_col_arrays = [image_data[var] for var in non_col_variables]

    # Stack arrays along the last dimension
    images_col = np.stack(col_arrays, axis=-1)
    images = np.stack(non_col_arrays, axis=-1)

    return images, images_col
