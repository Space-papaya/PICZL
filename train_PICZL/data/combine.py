import numpy as np
import os
import glob

# Change to your data directory
data_dir = '/home/wroster/learning-photoz/PICZL_OZ/train_PICZL/data/'
#os.chdir(data_dir)

# Find all _agn.npy files
agn_files = glob.glob('*_agn.npy')

for agn_file in agn_files:
	cosmos_file = agn_file.replace('_agn.npy', '_cosmos.npy')
	if not os.path.exists(cosmos_file):
		print(f"Warning: Cosmos file {cosmos_file} not found for {agn_file}")
		continue


	# Try loading with allow_pickle
	try:
		agn_data = np.load(agn_file, allow_pickle=True)
		cosmos_data = np.load(cosmos_file, allow_pickle=True)
	except Exception as e:
		print(f"Error loading {agn_file} or {cosmos_file}: {e}")
		continue

	# If it's a dict wrapped in .item(), unwrap it
	if isinstance(agn_data, np.ndarray) and agn_data.shape == ():
		agn_data = agn_data.item()
	if isinstance(cosmos_data, np.ndarray) and cosmos_data.shape == ():
		cosmos_data = cosmos_data.item()


	# Now handle based on type
	if isinstance(agn_data, np.ndarray) and isinstance(cosmos_data, np.ndarray):
		# Check shapes for array case
		if agn_data.shape[0] != cosmos_data.shape[0] or agn_data.shape[2:] != cosmos_data.shape[2:]:
			print(f"Shape mismatch between {agn_file} and {cosmos_file}, skipping.")
			continue

		combined_data = np.concatenate([agn_data, cosmos_data], axis=1)
		print(f'{agn_file}: {combined_data.shape}')

	elif isinstance(agn_data, dict) and isinstance(cosmos_data, dict):
		# For dicts, merge keys — assumes keys match
		combined_data = {}
		for key in agn_data:
			if key not in cosmos_data:
				print(f"Key {key} missing in cosmos data, skipping.")
				continue


			# Concatenate values under this key
			combined_data[key] = np.concatenate([agn_data[key], cosmos_data[key]], axis=0)
			print(f'{agn_file}: {key}, {combined_data[key].shape}')
	else:
		print(f"Unsupported data type in {agn_file}, skipping.")
		continue


	# Save combined — remove _agn suffix
	base_name = agn_file.replace('_agn.npy', '.npy')
	np.save(data_dir +"combined/" + base_name, combined_data)
	print(f"Saved combined file: {base_name}")
