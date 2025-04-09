

'''
#####  MODEL LOSS FUNCTIONS #####

This script holds custom redshift loss functions used for point estimates and pdfs. The latter introduces CRPS and NLL loss.

'''



import tensorflow as tf
from scipy.stats import norm
import numpy as np
import tensorflow_probability as tfp

#Definitions
tfd = tfp.distributions



# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------


#Point estimate loss function
def pe_loss(y_true, y_pred):
   error = abs(y_pred - y_true)/ (1+y_true)
   return tf.reduce_mean(error)


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------



#GMM - closed form CRPS for analytical solution when considering gaussians only

def A (means, sigmas):
    z = means / (sigmas + 1.0e-06)
    cdf = tfd.Normal(loc=0.0, scale=1.0).cdf(z)
    pdf = tfd.Normal(loc=0.0, scale=1.0).prob(z)
    return means * ((2 * cdf) - 1) + 2 * sigmas * pdf

def unpack(y_pred):
    # Split the concatenated parameters
    num_params = y_pred.shape[1]//3 #for integeer double slash
    means, sigmas, weights= tf.split(y_pred, 3, axis=1)
    return means, sigmas, weights, num_params


def crps_loss(y_true, y_pred):

    means, sigmas, weights, num_params = unpack(y_pred)
    # Repeat 'y_true' to match the shape of 'means' (batch_size, num_params)
    y_true_tiled = tf.tile(y_true, multiples=[1, num_params])

    crps_vector_batch = tf.reduce_sum(weights * A(y_true_tiled - means, sigmas), axis=1) - (0.5 * tf.reduce_sum(tf.expand_dims(weights,1) * tf.expand_dims(weights,2) * A((tf.expand_dims(means,1) - tf.expand_dims(means,2)),tf.sqrt(tf.expand_dims(sigmas**2,1)+tf.expand_dims(sigmas**2,2))), axis=(1,2)))
    return tf.reduce_mean((1.0/(1+y_true))*crps_vector_batch)


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------


#Define NLL loss

def NLL_loss(y_true, y_pred):

    means, sigmas, weights, num_params = unpack(y_pred)
    y_true_tiled = tf.tile(y_true, multiples=[1, num_params])

    # Define the PDF of the GMM
    pdf = tf.reduce_sum(weights * tfp.distributions.Normal(means, sigmas).prob(y_true_tiled), axis=1)

    # Calculate the NLL
    nll = -tf.math.log(pdf + 1e-10)  # Adding a small epsilon to avoid log(0)

    return tf.reduce_mean(nll) #added 1/(1+zs) factor


