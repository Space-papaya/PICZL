

'''
#####  PICZL CODE #####

This file holds function to train the CNN.

The catalogue on which these functions can be run requires following characteristics:

- Needs to follow the pre-processing performed in function "libs and data"
- ...

'''


from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import os


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------



def train_model(model, epochs, batch_size, learning_rate, loss_func, version, \
		train_images, train_col_images, train_features, train_labels, test_images, test_col_images, test_features, test_labels):

	'''
	Input: This function takes training hyperparameters as well as all training and test features as well as labels
	Output: Trained Model and training history for plotting

	This function trains the model on all relevant data.
	'''

	#set up reduced learning rate
	reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
	factor=0.9,
	patience=30,
	verbose=1,
	mode="auto",
	cooldown=0,
	min_lr=0.00001)

	early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
	min_delta=0.0003,
	patience=100)

	# Construct and create the checkpoint directory
	checkpoint_dir = os.path.expanduser(f'~/learning-photoz/PICZL_OZ/models/from_train/checkpoints/{version}')
	os.makedirs(checkpoint_dir, exist_ok=True)

	#Make sure to save the best model weights
	checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, '/best_model_weights.h5'), monitor='val_loss', save_best_only=True, mode='min', verbose=1)

	#Training about to start
	print('training about to start')

	#Train model and parse all neccessary variables, ####
	history = model.fit(x=[train_images, train_col_images, train_features], y=train_labels, epochs=epochs, callbacks=[checkpoint, early_stop, reduce_lr], batch_size=batch_size,\
				 validation_data = ([test_images, test_col_images, test_features], test_labels))

	#Training finsihed
	print('training finsihed')

	#Save the entire model
	save=0
	if save == True:
	        model.save('mymodel')


	## -----------------------------------------------------------------
	## -----------------------------------------------------------------


	#Load model weights from best weights saved
	best_weights = 1
	if best_weights == True:

		print('Loading best model weights')
		#Load best weights from Callback
		from tensorflow.keras.models import load_model

		#Compile the model with the custom loss function
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
			loss=loss_func,
			metrics=['accuracy'])

		#Load the best saved model weights
		model.load_weights(os.path.join(checkpoint_dir, '/best_model_weights.h5'))

	else:
		print('Not loading best model weights')

	return history, model


        # -------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------
