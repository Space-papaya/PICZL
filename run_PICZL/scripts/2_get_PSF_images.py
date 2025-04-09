
'''
2D Radially Symmetric Gaussian for PSF image
'''

import numpy as np
import pickle
import seaborn as sns
from astropy.table import Table
import matplotlib.pyplot as plt
from tqdm import tqdm

##############################

path = "/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/FLASH_30/"
ts = Table.read(path + 'FLASH_30_PICZL_ready.fits')
df = ts.to_pandas()

##############################


##############################

bands=['G','R','I','Z']
pixel_scale = 0.262
# Create a coordinate grid
size = 23  # Define the size of the image
x = np.linspace(-11, 11, 23) * pixel_scale
y = np.linspace(-11, 11, 23) * pixel_scale
x, y = np.meshgrid(x, y)
distance_squared = x**2 + y**2
gaussian_images = np.zeros((4, len(df), size, size))


# Generate Gaussian images for each row and band
for b, band in tqdm(enumerate(bands), total=len(bands), desc="Processing Bands"):
    for idx, row in df.iterrows():
        fwhm = row[f'PSFSIZE_{band}']
        sigma = fwhm / 2.355
        if sigma > 0:
            gaussian_images[b, idx] = (1/(2*np.pi*sigma**2))*np.exp(-(distance_squared) / (2 * sigma**2))
        else:
            gaussian_images[b, idx] = 0

# Check the shape of the resulting array
print(gaussian_images.shape)  # Should output (4, len(df), 23, 23)
##############################
store = "/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/FLASH_30/"
np.save(store + 'psf_images.npy', gaussian_images)
