# Training Hyperparameters
loss_func = crps_loss #NLL_loss  #crps_loss
epochs = 1000

batch_sizes = [1024,2048]
num_gaussian = [4,5,6,7]
learning_rates = [0.0003, 0.0005, 0.0007]
version = '0_1'
lf = 'CRPS'

# Initialize a list to store all training histories and configurations
all_histories_and_configs = []
all_predictions = []
all_train_predictions = []


model_counter=0
# Loop over hyperparameter values
for num_gauss in num_gaussian:
	for batch_size in batch_sizes:
		for learning_rate in learning_rates:

			model_counter = model_counter+1
			# Define a directory to save the models
			save_dir = "/home/wroster/learning-photoz/PICZL_OZ/train_PICZL/output/" + lf

			# Before starting a new model, clear the previous session:
			tf.keras.backend.clear_session()

			# Create and train multiple models
			model = compile_model(catalog_features.shape[1], num_gauss, learning_rate, loss_func)
			history, model = train_model(model, epochs, batch_size, learning_rate, loss_func,
						train_images, train_col_images, train_features, train_labels, test_images, test_col_images, test_features, test_labels)

			print(f"Model {model_counter} trained. Validation Loss: {min(history.history['val_loss'])}")

			# Save the model to a file
			model_file = os.path.join(save_dir + "/models/" + version, f"model_G={num_gauss}_B={batch_size}_lr={learning_rate}.h5")
			model.save(model_file)
			preds = model.predict([test_images, test_col_images, test_features])
			#np.save(os.path.join(save_dir + "/preds/" + version, f"preds_G={num_gauss}_B={batch_size}_lr={learning_rate}.npy"), preds)
			all_predictions.append(preds)

			# Save the training history and configurations
			config = {'gmm_components': num_gauss, 'batch_size': batch_size, 'learning_rate': learning_rate}
			history_and_config = {'config': config, 'history': history.history}
			#with open(os.path.join(save_dir + "/hists/" + version, f"history_G={num_gauss}_B={batch_size}_lr={learning_rate}.pkl"), 'wb') as f:
			#	pickle.dump(history.history, f)
			all_histories_and_configs.append(history_and_config)

			# After model is saved and predictions done:
			del model
			K.clear_session()
			gc.collect()


return all_predictions, all_histories_and_configs


