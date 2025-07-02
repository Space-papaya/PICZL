#

'''
#####  CNN MODEL #####

This file holds functions on the nature of the CNN.

The variables on which these functions can be run requires following characteristics:

        - Need to fit with the defined paramater dimensions in the network
        - ...

'''



##Import the libraries
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Conv2D,Dropout,MaxPooling2D,Flatten, Conv1D, BatchNormalization, Concatenate, MaxPooling1D, AveragePooling2D, UpSampling2D, ZeroPadding2D, Reshape, Lambda
from keras.layers import RandomRotation, RandomFlip
from tensorflow.keras.utils import plot_model


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------


def compile_model(non_2D_shape, lastlayerdimension, learn_rate, loss_func):

        """
        Input: Takes the number of gaussian components which defines the last layer dimension
        Output: Returns Model architecture

        CNN with multiple inputs: cutout images, color images, numerical features, aperture photometry. This network delivers the means, standard deviations and weights for a gaus>
        """


        # Define the input layers for the two inputs
        input_layer_1 = Input(shape=(23, 23, 36)) #should be 32, 36 with PSF
        input_layer_12 = Input(shape=(23, 23, 24)) #should be 24
        input_layer_6 = Input(shape=(non_2D_shape,))


        # Define shared hyperparameters
        filter_size=32
        kernel_size = (5, 5)
        kernel_size_small = (4, 4)
        activation_comb = 'sigmoid'
        activation = 'softmax'
        padding = 'same'
        k_nodes = 100
        l_nodes = 75
        fc_nodes1 = 120
        fc_nodes2 = 90
        fc_nodes3 = 100

        # Branch 1, images
        a = Conv2D(filter_size, kernel_size=kernel_size, activation=activation_comb, padding=padding)(input_layer_1)
        a = MaxPooling2D(pool_size=(2, 2))(a)
        a = Dropout(0.28)(a)
        a = Conv2D(2 * filter_size, kernel_size=kernel_size_small, activation=activation_comb, padding=padding)(a)
        a = MaxPooling2D(pool_size=(2, 2))(a)
        a = Conv2D(4 * filter_size, kernel_size=kernel_size_small, activation=activation_comb, padding=padding)(a)
        a = Conv2D(2 * filter_size, kernel_size=(1,1), activation=activation_comb, padding=padding)(a)
        a = Flatten()(a)
        a = Dense(fc_nodes1, activation=activation_comb)(a)
        a = Dropout(0.33)(a)
        for _ in range(2):
                a = Dense(fc_nodes2, activation=activation_comb)(a)

        a = Dense(fc_nodes3, activation='linear')(a)



        # Branch 2, colour images
        j = Conv2D(filter_size, kernel_size=kernel_size, activation=activation, padding=padding)(input_layer_12)
        j = MaxPooling2D(pool_size=(2, 2))(j)
        j = Dropout(0.38)(j)
        j = Conv2D(2 * filter_size, kernel_size=kernel_size_small, activation=activation, padding=padding)(j)
        j = MaxPooling2D(pool_size=(2, 2))(j)
        j = Conv2D(4 * filter_size, kernel_size=kernel_size_small, activation=activation, padding=padding)(j)
        j = Conv2D(2 * filter_size, kernel_size=(1,1), activation=activation, padding=padding)(j)
        j = Flatten()(j)
        j = Dense(fc_nodes1, activation=activation)(j)
        j = Dropout(0.35)(j)
        for _ in range(2):
                j = Dense(fc_nodes2, activation=activation)(j)

        j = Dense(fc_nodes3, activation='linear')(j)


        # Branch 3, non 2D features
        k = Dense(k_nodes, activation=activation_comb)(input_layer_6)
        for _ in range(5):
                k = Dense(k_nodes, activation=activation_comb)(k)
        k = Dropout(0.4)(k)
        for _ in range(7):
                k = Dense(k_nodes, activation=activation_comb)(k)


        # Concatenate both branches
        l = Concatenate(axis=-1)([a,j,k])
        for _ in range(3):
                l = Dense(l_nodes)(l)
        l = Dropout(0.4)(l)
        for _ in range(3):
                l = Dense(l_nodes)(l)



        # Add GMM-related layers
        means = Dense(lastlayerdimension, activation='relu')(l)
        stds = Dense(lastlayerdimension, activation='softplus')(l)
        weights = Dense(lastlayerdimension, activation='softmax')(l)

        # Concatenate the GMM parameters
        gmm_params = Concatenate(axis=1)([means, stds, weights])

        # Create the model
        model = Model(inputs= [input_layer_1, input_layer_12, input_layer_6], outputs=gmm_params)

        #compile the model with a learning rate and custom loss function
        model.compile(optimizer=tf.keras.optimizers.Adam(learn_rate, clipvalue=1.0),
              loss=loss_func,
              metrics=['accuracy'])

        #adding text description of current training checkpoint
        print('>> Model compiled')

        return model

        # -------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------

