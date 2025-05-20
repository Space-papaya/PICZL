
import numpy as np

def stack_images(image_data):

	# Separate variables based on whether they end with "col"
	col_variables = [var_name for var_name in image_data.keys() if var_name.endswith("col")]
	non_col_variables = [var_name for var_name in image_data.keys() if not var_name.endswith("col")]

	# Extract their values directly from image_data
	col_arrays = [image_data[var] for var in col_variables]
	non_col_arrays = [image_data[var] for var in non_col_variables]

	'''
	print("Col arrays:")
	for i in range(len(col_arrays)):
		print(f"{col_variables[i]} : {col_arrays[i].shape}")

	print("\nNon-col arrays:")
	for i in range(len(non_col_arrays)):
		print(f"{non_col_variables[i]} : {non_col_arrays[i].shape}")


	'''
	# Stack along the last axis
	images_col = np.stack(col_arrays, axis=-1)
	images = np.stack(non_col_arrays, axis=-1)

	return images, images_col

