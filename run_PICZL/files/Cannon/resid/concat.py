import numpy as np
import glob
import re

# Step 1: Get list of all files
files = glob.glob('dered_griz_resid_*')

# Step 2: Extract numerical keys for sorting
def extract_key(filename):
    match = re.search(r'(\d+)_(\d+)', filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    else:
        return (float('inf'), float('inf'))  # fallback to push invalids to end

# Step 3: Sort files based on your required order
files_sorted = sorted(files, key=extract_key)

print(f"Found {len(files_sorted)} files, first few sorted names:")
for f in files_sorted[:5]:
    print(f)

# Step 4: Load and concatenate
all_arrays = []

for f in files_sorted:
    data = np.load(f)  # assuming .npy files; adjust if needed
    if data.shape != (4, 5, 23, 23):
        print(f"Warning: unexpected shape {data.shape} in file {f}")
    all_arrays.append(data)

# Concatenate along axis 1 (sources)
concatenated = np.concatenate(all_arrays, axis=1)

print(f"Final concatenated shape: {concatenated.shape}")

# Optional: save result
path='/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/Cannon/'
np.save(path+ 'dered_griz_resid.npy', concatenated)
