import tensorflow as tf
from scipy.stats import norm
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions


def pe_loss(y_true, y_pred):
    """
    Point estimate loss function.

    Computes the normalized absolute error between predicted and true values:
    |z_pred - z_true| / (1 + z_true)

    Parameters:
        y_true (tf.Tensor): Ground truth redshift values of shape (batch_size, 1)
        y_pred (tf.Tensor): Predicted redshift values of shape (batch_size, 1)

    Returns:
        tf.Tensor: Mean normalized absolute error
    """
    error = abs(y_pred - y_true) / (1 + y_true)
    return tf.reduce_mean(error)



def A(means, sigmas):
    """
    Helper function for CRPS loss that evaluates an analytical component of the score.

    Parameters:
        means (tf.Tensor): Difference between true value and predicted means (shape: [batch_size, num_components])
        sigmas (tf.Tensor): Standard deviations of Gaussian components (same shape as means)

    Returns:
        tf.Tensor: Computed A value for CRPS
    """
    z = means / (sigmas + 1.0e-06)
    cdf = tfd.Normal(loc=0.0, scale=1.0).cdf(z)
    pdf = tfd.Normal(loc=0.0, scale=1.0).prob(z)
    return means * ((2 * cdf) - 1) + 2 * sigmas * pdf



def unpack(y_pred):
    """
    Unpacks the concatenated GMM parameters from the model output.

    Parameters:
        y_pred (tf.Tensor): Tensor of shape (batch_size, 3 * num_components) containing
                            concatenated means, sigmas, and weights.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, int]:
            - means: predicted means of shape (batch_size, num_components)
            - sigmas: predicted standard deviations
            - weights: predicted component weights
            - num_params: number of Gaussian components
    """
    num_params = y_pred.shape[1] // 3
    means, sigmas, weights = tf.split(y_pred, 3, axis=1)
    return means, sigmas, weights, num_params


def crps_loss(y_true, y_pred):
    """
    Computes the Continuous Ranked Probability Score (CRPS) loss for Gaussian Mixture Models.

    Parameters:
        y_true (tf.Tensor): Ground truth redshift values (batch_size, 1)
        y_pred (tf.Tensor): Model outputs containing concatenated GMM parameters

    Returns:
        tf.Tensor: CRPS loss averaged over the batch
    """
    means, sigmas, weights, num_params = unpack(y_pred)
    y_true_tiled = tf.tile(y_true, multiples=[1, num_params])

    crps_vector_batch = tf.reduce_sum(
        weights * A(y_true_tiled - means, sigmas), axis=1
    ) - (
        0.5
        * tf.reduce_sum(
            tf.expand_dims(weights, 1)
            * tf.expand_dims(weights, 2)
            * A(
                tf.expand_dims(means, 1) - tf.expand_dims(means, 2),
                tf.sqrt(tf.expand_dims(sigmas**2, 1) + tf.expand_dims(sigmas**2, 2)),
            ),
            axis=(1, 2),
        )
    )

    return tf.reduce_mean((1.0 / (1 + y_true)) * crps_vector_batch)



def NLL_loss(y_true, y_pred):
    """
    Computes the Negative Log Likelihood (NLL) loss for a Gaussian Mixture Model.

    Parameters:
        y_true (tf.Tensor): Ground truth redshift values (batch_size, 1)
        y_pred (tf.Tensor): Model outputs containing concatenated GMM parameters

    Returns:
        tf.Tensor: NLL loss averaged over the batch
    """
    means, sigmas, weights, num_params = unpack(y_pred)
    y_true_tiled = tf.tile(y_true, multiples=[1, num_params])

    pdf = tf.reduce_sum(
        weights * tfp.distributions.Normal(means, sigmas).prob(y_true_tiled),
        axis=1
    )

    nll = -tf.math.log(pdf + 1e-10)
    return tf.reduce_mean(nll)
