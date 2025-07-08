
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Dropout, MaxPooling2D, Flatten,
    Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

def compile_model(non_2D_shape, lastlayerdimension, learn_rate, loss_func):
    """
    Builds and compiles a multi-input CNN model for computing the parameters of a
    Gaussian Mixture Model (GMM). The network is designed to ingest three types
    of inputs:
        1. PSF-stacked cutout images
        2. Color cutout images
        3. Non-image numerical features (e.g., morphology)

    Parameters
    ----------
    non_2D_shape : int
        The number of 1D input features (e.g., number of photometric bands).
    lastlayerdimension : int
        The number of GMM components × 3 (for mean, std, weight per component).
    learn_rate : float
        The learning rate for the Adam optimizer.
    loss_func : callable
        The custom loss function for training.

    Returns
    -------
    tf.keras.Model
        A compiled Keras model ready for training.
    """

    # Input layers
    input_layer_1 = Input(shape=(23, 23, 36), name="psf_input")
    input_layer_12 = Input(shape=(23, 23, 24), name="color_input")
    input_layer_6 = Input(shape=(non_2D_shape,), name="feature_input")

    # Shared architectural constants
    filter_size = 32
    kernel_size = (5, 5)
    kernel_size_small = (4, 4)
    activation_comb = "sigmoid"
    activation = "softmax"
    padding = "same"
    k_nodes = 100
    l_nodes = 75
    fc_nodes1 = 120
    fc_nodes2 = 90
    fc_nodes3 = 100

    # ------------------------------------------------------------------------------
    # Branch 1: PSF image input
    a = Conv2D(filter_size, kernel_size, activation=activation_comb, padding=padding)(input_layer_1)
    a = MaxPooling2D(pool_size=(2, 2))(a)
    a = Dropout(0.28)(a)

    a = Conv2D(2 * filter_size, kernel_size_small, activation=activation_comb, padding=padding)(a)
    a = MaxPooling2D(pool_size=(2, 2))(a)

    a = Conv2D(4 * filter_size, kernel_size_small, activation=activation_comb, padding=padding)(a)
    a = Conv2D(2 * filter_size, (1, 1), activation=activation_comb, padding=padding)(a)

    a = Flatten()(a)
    a = Dense(fc_nodes1, activation=activation_comb)(a)
    a = Dropout(0.33)(a)

    for _ in range(2):
        a = Dense(fc_nodes2, activation=activation_comb)(a)

    a = Dense(fc_nodes3, activation="linear")(a)

    # ------------------------------------------------------------------------------
    # Branch 2: Color image input
    j = Conv2D(filter_size, kernel_size, activation=activation, padding=padding)(input_layer_12)
    j = MaxPooling2D(pool_size=(2, 2))(j)
    j = Dropout(0.38)(j)

    j = Conv2D(2 * filter_size, kernel_size_small, activation=activation, padding=padding)(j)
    j = MaxPooling2D(pool_size=(2, 2))(j)

    j = Conv2D(4 * filter_size, kernel_size_small, activation=activation, padding=padding)(j)
    j = Conv2D(2 * filter_size, (1, 1), activation=activation, padding=padding)(j)

    j = Flatten()(j)
    j = Dense(fc_nodes1, activation=activation)(j)
    j = Dropout(0.35)(j)

    for _ in range(2):
        j = Dense(fc_nodes2, activation=activation)(j)

    j = Dense(fc_nodes3, activation="linear")(j)

    # ------------------------------------------------------------------------------
    # Branch 3: 1D features input
    k = Dense(k_nodes, activation=activation_comb)(input_layer_6)
    for _ in range(5):
        k = Dense(k_nodes, activation=activation_comb)(k)

    k = Dropout(0.4)(k)
    for _ in range(7):
        k = Dense(k_nodes, activation=activation_comb)(k)

    # ------------------------------------------------------------------------------
    # Merge and Output
    l = Concatenate(axis=-1)([a, j, k])

    for _ in range(3):
        l = Dense(l_nodes)(l)
    l = Dropout(0.4)(l)
    for _ in range(3):
        l = Dense(l_nodes)(l)

    # Final output: GMM parameters
    means = Dense(lastlayerdimension, activation="relu", name="means")(l)
    stds = Dense(lastlayerdimension, activation="softplus", name="stds")(l)
    weights = Dense(lastlayerdimension, activation="softmax", name="weights")(l)

    gmm_params = Concatenate(axis=1, name="gmm_output")([means, stds, weights])

    # Build and compile the model
    model = Model(inputs=[input_layer_1, input_layer_12, input_layer_6], outputs=gmm_params)

    model.compile(
        optimizer=Adam(learn_rate, clipvalue=1.0),
        loss=loss_func,
        metrics=["accuracy"]
    )

    print(">> Model compiled")

    return model
