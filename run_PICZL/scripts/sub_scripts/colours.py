
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from tqdm import tqdm


##################################################################
##################################################################


@jit
# Define pixel_threshold function
def pixel_threshold(image, threshold):
	pixel_noise = np.random.uniform(0.00001, 0.00005)
	return jnp.where(image > threshold, image, pixel_noise)


##################################################################


def clean_obs(images_g, images_r, images_i, images_z):


	for i in tqdm(range(0, len(images_g))):
		for j in range(0,23):
			for k in range(0,23):
				#corner and rim
				if j<1: #upper corners
					if k<1: #top left
						pixels_edge_g = images_g[i][j][k+1], images_g[i][j+1][k], images_g[i][j+1][k+1]
						pixels_edge_r = images_r[i][j][k+1], images_r[i][j+1][k], images_r[i][j+1][k+1]
						pixels_edge_i = images_i[i][j][k+1], images_i[i][j+1][k], images_i[i][j+1][k+1]
						pixels_edge_z = images_z[i][j][k+1], images_z[i][j+1][k], images_z[i][j+1][k+1]
					if k>21: #top right
						pixels_edge_g = images_g[i][j][k-1], images_g[i][j+1][k], images_g[i][j+1][k-1]
						pixels_edge_r = images_r[i][j][k-1], images_r[i][j+1][k], images_r[i][j+1][k-1]
						pixels_edge_i = images_i[i][j][k-1], images_i[i][j+1][k], images_i[i][j+1][k-1]
						pixels_edge_z = images_z[i][j][k-1], images_z[i][j+1][k], images_z[i][j+1][k-1]
					if k>0 and k<22: #top side
						pixels_edge_g = images_g[i][j][k-1], images_g[i][j+1][k-1], images_g[i][j+1][k], images_g[i][j+1][k+1], images_g[i][j][k+1]
						pixels_edge_r = images_r[i][j][k-1], images_r[i][j+1][k-1], images_r[i][j+1][k], images_r[i][j+1][k+1], images_r[i][j][k+1]
						pixels_edge_i = images_i[i][j][k-1], images_i[i][j+1][k-1], images_i[i][j+1][k], images_i[i][j+1][k+1], images_i[i][j][k+1]
						pixels_edge_z = images_z[i][j][k-1], images_z[i][j+1][k-1], images_z[i][j+1][k], images_z[i][j+1][k+1], images_z[i][j][k+1]

					if (images_g[i][j][k]/np.mean(pixels_edge_g)) > 3 or (images_g[i][j][k]/np.mean(pixels_edge_g)) < 1/3:
						images_g[i][j][k] = np.mean(pixels_edge_g)
					if (images_r[i][j][k]/np.mean(pixels_edge_r)) > 3 or (images_r[i][j][k]/np.mean(pixels_edge_r)) < 1/3:
						images_r[i][j][k] = np.mean(pixels_edge_r)
					if (images_i[i][j][k]/np.mean(pixels_edge_i)) > 3 or (images_i[i][j][k]/np.mean(pixels_edge_i)) < 1/3:
						images_i[i][j][k] = np.mean(pixels_edge_i)
					if (images_z[i][j][k]/np.mean(pixels_edge_z)) > 3 or (images_z[i][j][k]/np.mean(pixels_edge_z)) < 1/3:
						images_z[i][j][k] = np.mean(pixels_edge_z)




	for i in tqdm(range(0, len(images_g))):
		for j in range(0,23):
			for k in range(0,23):
				#left side right side
				if j>0 and j<22 and (k<1 or k>21):
					if k<1: #left side
						pixels_edge_g = images_g[i][j-1][k], images_g[i][j-1][k+1], images_g[i][j][k+1], images_g[i][j+1][k+1], images_g[i][j+1][k]
						pixels_edge_r = images_r[i][j-1][k], images_r[i][j-1][k+1], images_r[i][j][k+1], images_r[i][j+1][k+1], images_r[i][j+1][k]
						pixels_edge_i = images_i[i][j-1][k], images_i[i][j-1][k+1], images_i[i][j][k+1], images_i[i][j+1][k+1], images_i[i][j+1][k]
						pixels_edge_z = images_z[i][j-1][k], images_z[i][j-1][k+1], images_z[i][j][k+1], images_z[i][j+1][k+1], images_z[i][j+1][k]
					if k>21: #right side
						pixels_edge_g = images_g[i][j-1][k], images_g[i][j-1][k-1], images_g[i][j][k-1], images_g[i][j+1][k-1], images_g[i][j+1][k]
						pixels_edge_r = images_r[i][j-1][k], images_r[i][j-1][k-1], images_r[i][j][k-1], images_r[i][j+1][k-1], images_r[i][j+1][k]
						pixels_edge_i = images_i[i][j-1][k], images_i[i][j-1][k-1], images_i[i][j][k-1], images_i[i][j+1][k-1], images_i[i][j+1][k]
						pixels_edge_z = images_z[i][j-1][k], images_z[i][j-1][k-1], images_z[i][j][k-1], images_z[i][j+1][k-1], images_z[i][j+1][k]


					if (images_g[i][j][k]/np.mean(pixels_edge_g)) > 3 or (images_g[i][j][k]/np.mean(pixels_edge_g)) < 1/3:
						images_g[i][j][k] = np.mean(pixels_edge_g)
					if (images_r[i][j][k]/np.mean(pixels_edge_r)) > 3 or (images_r[i][j][k]/np.mean(pixels_edge_r)) < 1/3:
						images_r[i][j][k] = np.mean(pixels_edge_r)
					if (images_i[i][j][k]/np.mean(pixels_edge_i)) > 3 or (images_i[i][j][k]/np.mean(pixels_edge_i)) < 1/3:
						images_i[i][j][k] = np.mean(pixels_edge_i)
					if (images_z[i][j][k]/np.mean(pixels_edge_z)) > 3 or (images_z[i][j][k]/np.mean(pixels_edge_z)) < 1/3:
						images_z[i][j][k] = np.mean(pixels_edge_z)



	for i in tqdm(range(0, len(images_g))):
		for j in range(1,22):
			for k in range(1,22):
				#main body
				if j>0 and j<22 and k>0 and k<22:
					pixels_g = [images_g[i][j-1][k-1], images_g[i][j-1][k], images_g[i][j-1][k+1], images_g[i][j][k-1], images_g[i][j][k+1], images_g[i][j+1][k-1], images_g[i][j+1][k], images_g[i][j+1][k+1]]  
					pixels_r = [images_r[i][j-1][k-1], images_r[i][j-1][k], images_r[i][j-1][k+1], images_r[i][j][k-1], images_r[i][j][k+1], images_r[i][j+1][k-1], images_r[i][j+1][k], images_r[i][j+1][k+1]]
					pixels_i = [images_i[i][j-1][k-1], images_i[i][j-1][k], images_i[i][j-1][k+1], images_i[i][j][k-1], images_i[i][j][k+1], images_i[i][j+1][k-1], images_i[i][j+1][k], images_i[i][j+1][k+1]]
					pixels_z = [images_z[i][j-1][k-1], images_z[i][j-1][k], images_z[i][j-1][k+1], images_z[i][j][k-1], images_z[i][j][k+1], images_z[i][j+1][k-1], images_z[i][j+1][k], images_z[i][j+1][k+1]]

				if (images_g[i][j][k]/np.median(pixels_g)) > 3 or (images_g[i][j][k]/np.mean(pixels_g)) < 1/3:
					images_g[i][j][k] = np.median(pixels_g)
				if (images_r[i][j][k]/np.median(pixels_r)) > 3 or (images_r[i][j][k]/np.mean(pixels_r)) < 1/3:
					images_r[i][j][k] = np.median(pixels_r)
				if (images_i[i][j][k]/np.median(pixels_i)) > 3 or (images_i[i][j][k]/np.mean(pixels_i)) < 1/3:
					images_i[i][j][k] = np.median(pixels_i)
				if (images_z[i][j][k]/np.median(pixels_z)) > 3 or (images_z[i][j][k]/np.mean(pixels_z)) < 1/3:
					images_z[i][j][k] = np.median(pixels_z)

	#np.save('/home/wroster/scripts/data/diff/new/g_after_noise_b.npy', images_g)


	for i in tqdm(range(0, len(images_g))):
		for j in range(0,23):
			for k in range(0,23):
				if j>21: #lower corners
					if k<1: #lower left
						pixels_edge_g = images_g[i][j][k+1], images_g[i][j-1][k], images_g[i][j-1][k+1]
						pixels_edge_r = images_r[i][j][k+1], images_r[i][j-1][k], images_r[i][j-1][k+1]
						pixels_edge_i = images_i[i][j][k+1], images_i[i][j-1][k], images_i[i][j-1][k+1]
						pixels_edge_z = images_z[i][j][k+1], images_z[i][j-1][k], images_z[i][j-1][k+1]
					if k>21: #lower right
						pixels_edge_g = images_g[i][j][k-1], images_g[i][j-1][k], images_g[i][j-1][k-1]
						pixels_edge_r = images_r[i][j][k-1], images_r[i][j-1][k], images_r[i][j-1][k-1]
						pixels_edge_i = images_i[i][j][k-1], images_i[i][j-1][k], images_i[i][j-1][k-1]
						pixels_edge_z = images_z[i][j][k-1], images_z[i][j-1][k], images_z[i][j-1][k-1]
					if k>0 and k<22: #lower side
						pixels_edge_g = images_g[i][j][k-1], images_g[i][j-1][k-1], images_g[i][j-1][k], images_g[i][j-1][k+1], images_g[i][j][k+1]
						pixels_edge_r = images_r[i][j][k-1], images_r[i][j-1][k-1], images_r[i][j-1][k], images_r[i][j-1][k+1], images_r[i][j][k+1]
						pixels_edge_i = images_i[i][j][k-1], images_i[i][j-1][k-1], images_i[i][j-1][k], images_i[i][j-1][k+1], images_i[i][j][k+1]
						pixels_edge_z = images_z[i][j][k-1], images_z[i][j-1][k-1], images_z[i][j-1][k], images_z[i][j-1][k+1], images_z[i][j][k+1]

					if (images_g[i][j][k]/np.mean(pixels_edge_g)) > 3 or (images_g[i][j][k]/np.mean(pixels_edge_g)) < 1/3:
						images_g[i][j][k] = np.mean(pixels_edge_g)
					if (images_r[i][j][k]/np.mean(pixels_edge_r)) > 3 or (images_r[i][j][k]/np.mean(pixels_edge_r)) < 1/3:
						images_r[i][j][k] = np.mean(pixels_edge_r)
					if (images_i[i][j][k]/np.mean(pixels_edge_i)) > 3 or (images_i[i][j][k]/np.mean(pixels_edge_i)) < 1/3:
						images_i[i][j][k] = np.mean(pixels_edge_i)
					if (images_z[i][j][k]/np.mean(pixels_edge_z)) > 3 or (images_z[i][j][k]/np.mean(pixels_edge_z)) < 1/3:
						images_z[i][j][k] = np.mean(pixels_edge_z)

	#np.save('/home/wroster/scripts/data/diff/new/g_after_noise_a.npy', images_g)


	#For images which only have noise, set them to zero as non-detection
	limit = 0.0001
	default_mag = 0

	if np.max(images_g[i]) < limit:
		for j in range(0,23):
			for k in range(0,23):
				images_g[i][j][k] = default_mag

	if np.max(images_r[i]) < limit:
		for j in range(0,23):
			for k in range(0,23):
				images_r[i][j][k] = default_mag

	if np.max(images_i[i]) < limit:
		for j in range(0,23):
			for k in range(0,23):
				images_i[i][j][k] = default_mag

	if np.max(images_z[i]) < limit:
		for j in range(0,23):
			for k in range(0,23):
				images_z[i][j][k] = default_mag



	clean_griz = np.stack((images_g, images_r, images_i, images_z), axis=0)
	np.save('/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/FLASH_30/clean_dered_griz.npy', clean_griz)


##################################################################

def clean_cols(images_g, images_r, images_i, images_z):


	images_gr=np.zeros((len(images_g),23,23))
	images_gi=np.zeros((len(images_g),23,23))
	images_gz=np.zeros((len(images_g),23,23))
	images_ri=np.zeros((len(images_g),23,23))
	images_rz=np.zeros((len(images_g),23,23))
	images_iz=np.zeros((len(images_g),23,23))

	default = -99


	for i in tqdm(range(0, len(images_g))):
		#g-r
		if (np.max(images_g[i]) == 0) |  (np.max(images_r[i]) == 0):
			for j in range(0,23):
				for k in range(0,23):
					images_gr[i][j][k] = default
		else:
			for j in range(0,23):
				for k in range(0,23):
					images_gr[i][j][k] = (22.5 - (2.5*(np.log10(images_g[i][j][k])))) - (22.5 - (2.5*(np.log10(images_r[i][j][k]))))


		#g-i
		if (np.max(images_g[i]) == 0) |  (np.max(images_i[i]) == 0):
			for j in range(0,23):
				for k in range(0,23):
					images_gi[i][j][k] = default
		else:
			for j in range(0,23):
				for k in range(0,23):
					images_gi[i][j][k] = (22.5 - (2.5*(np.log10(images_g[i][j][k])))) - (22.5 - (2.5*(np.log10(images_i[i][j][k]))))


		#g-z
		if (np.max(images_g[i]) == 0) |  (np.max(images_z[i]) == 0):
			for j in range(0,23):
				for k in range(0,23):
					images_gz[i][j][k] = default
		else:
			for j in range(0,23):
				for k in range(0,23):
					images_gz[i][j][k] = (22.5 - (2.5*(np.log10(images_g[i][j][k])))) - (22.5 - (2.5*(np.log10(images_z[i][j][k]))))


		#r-i
		if (np.max(images_r[i]) == 0) |  (np.max(images_i[i]) == 0):
			for j in range(0,23):
				for k in range(0,23):
					images_ri[i][j][k] = default
		else:
			for j in range(0,23):
				for k in range(0,23):
					images_ri[i][j][k] = (22.5 - (2.5*(np.log10(images_r[i][j][k])))) - (22.5 - (2.5*(np.log10(images_i[i][j][k]))))


		#r-z
		if (np.max(images_r[i]) == 0) |  (np.max(images_z[i]) == 0):
			for j in range(0,23):
				for k in range(0,23):
					images_rz[i][j][k] = default
		else:
			for j in range(0,23):
				for k in range(0,23):
					images_rz[i][j][k] = (22.5 - (2.5*(np.log10(images_r[i][j][k])))) - (22.5 - (2.5*(np.log10(images_z[i][j][k]))))


		#i-z
		if (np.max(images_i[i]) == 0) |  (np.max(images_z[i]) == 0):
			for j in range(0,23):
				for k in range(0,23):
					images_iz[i][j][k] = default
		else:
			for j in range(0,23):
				for k in range(0,23):
					images_iz[i][j][k] = (22.5 - (2.5*(np.log10(images_i[i][j][k])))) - (22.5 - (2.5*(np.log10(images_z[i][j][k]))))




	for i in tqdm(range(0, len(images_g))):
		for j in range(1,22):
			for k in range(1,22):
				if j>0 and j<22 and k>0 and k<22:
					#main body
					pixels_gr = [images_gr[i][j-1][k-1], images_gr[i][j-1][k], images_gr[i][j-1][k+1], images_gr[i][j][k-1], images_gr[i][j][k+1], images_gr[i][j+1][k-1], images_gr[i][j+1][k], images_gr[i][j+1][k+1]]  
					pixels_gi = [images_gi[i][j-1][k-1], images_gi[i][j-1][k], images_gi[i][j-1][k+1], images_gi[i][j][k-1], images_gi[i][j][k+1], images_gi[i][j+1][k-1], images_gi[i][j+1][k], images_gi[i][j+1][k+1]]
					pixels_gz = [images_gz[i][j-1][k-1], images_gz[i][j-1][k], images_gz[i][j-1][k+1], images_gz[i][j][k-1], images_gz[i][j][k+1], images_gz[i][j+1][k-1], images_gz[i][j+1][k], images_gz[i][j+1][k+1]]
					pixels_ri = [images_ri[i][j-1][k-1], images_ri[i][j-1][k], images_ri[i][j-1][k+1], images_ri[i][j][k-1], images_ri[i][j][k+1], images_ri[i][j+1][k-1], images_ri[i][j+1][k], images_ri[i][j+1][k+1]]
					pixels_rz = [images_rz[i][j-1][k-1], images_rz[i][j-1][k], images_rz[i][j-1][k+1], images_rz[i][j][k-1], images_rz[i][j][k+1], images_rz[i][j+1][k-1], images_rz[i][j+1][k], images_rz[i][j+1][k+1]]
					pixels_iz = [images_iz[i][j-1][k-1], images_iz[i][j-1][k], images_iz[i][j-1][k+1], images_iz[i][j][k-1], images_iz[i][j][k+1], images_iz[i][j+1][k-1], images_iz[i][j+1][k], images_iz[i][j+1][k+1]]


				if (np.max(images_gr[i][j][k]) > -99):
					if (abs(images_gr[i][j][k]) - abs(np.median(pixels_gr))) > 0.3:
						images_gr[i][j][k] = np.median(pixels_gr)
				if (np.max(images_gi[i][j][k]) > -99):
					if (abs(images_gi[i][j][k]) - abs(np.median(pixels_gi))) > 0.3:
						images_gi[i][j][k] = np.median(pixels_gi)
				if (np.max(images_gz[i][j][k]) > -99):
					if (abs(images_gz[i][j][k]) - abs(np.median(pixels_gz))) > 0.3:
						images_gz[i][j][k] = np.median(pixels_gz)
				if (np.max(images_ri[i][j][k]) > -99):
					if (abs(images_ri[i][j][k]) - abs(np.median(pixels_ri))) > 0.3:
						images_ri[i][j][k] = np.median(pixels_ri)
				if (np.max(images_rz[i][j][k]) > -99):
					if (abs(images_rz[i][j][k]) - abs(np.median(pixels_rz))) > 0.3:
						images_rz[i][j][k] = np.median(pixels_rz)
				if (np.max(images_iz[i][j][k]) > -99):
					if (abs(images_iz[i][j][k]) - abs(np.median(pixels_iz))) > 0.3:
						images_iz[i][j][k] = np.median(pixels_iz)


	for i in tqdm(range(0, len(images_g))):
		for j in range(0,23):
			for k in range(0,23):
			#corner and rim
				if j<1: #upper corners
					if k<1: #top left
						pixels_edge_gr = images_gr[i][j][k+1], images_gr[i][j+1][k], images_gr[i][j+1][k+1]
						pixels_edge_gi = images_gi[i][j][k+1], images_gi[i][j+1][k], images_gi[i][j+1][k+1]
						pixels_edge_gz = images_gz[i][j][k+1], images_gz[i][j+1][k], images_gz[i][j+1][k+1]
						pixels_edge_ri = images_ri[i][j][k+1], images_ri[i][j+1][k], images_ri[i][j+1][k+1]
						pixels_edge_rz = images_rz[i][j][k+1], images_rz[i][j+1][k], images_rz[i][j+1][k+1]
						pixels_edge_iz = images_iz[i][j][k+1], images_iz[i][j+1][k], images_iz[i][j+1][k+1]

					if k>21: #top right
						pixels_edge_gr = images_gr[i][j][k-1], images_gr[i][j+1][k], images_gr[i][j+1][k-1]
						pixels_edge_gi = images_gi[i][j][k-1], images_gi[i][j+1][k], images_gi[i][j+1][k-1]
						pixels_edge_gz = images_gz[i][j][k-1], images_gz[i][j+1][k], images_gz[i][j+1][k-1]
						pixels_edge_ri = images_ri[i][j][k-1], images_ri[i][j+1][k], images_ri[i][j+1][k-1]
						pixels_edge_rz = images_rz[i][j][k-1], images_rz[i][j+1][k], images_rz[i][j+1][k-1]
						pixels_edge_iz = images_iz[i][j][k-1], images_iz[i][j+1][k], images_iz[i][j+1][k-1]

					if k>0 and k<22: #top side
						pixels_edge_gr = images_gr[i][j][k-1], images_gr[i][j+1][k-1], images_gr[i][j+1][k], images_gr[i][j+1][k+1], images_gr[i][j][k+1]
						pixels_edge_gi = images_gi[i][j][k-1], images_gi[i][j+1][k-1], images_gi[i][j+1][k], images_gi[i][j+1][k+1], images_gi[i][j][k+1]
						pixels_edge_gz = images_gz[i][j][k-1], images_gz[i][j+1][k-1], images_gz[i][j+1][k], images_gz[i][j+1][k+1], images_gz[i][j][k+1]
						pixels_edge_ri = images_ri[i][j][k-1], images_ri[i][j+1][k-1], images_ri[i][j+1][k], images_ri[i][j+1][k+1], images_ri[i][j][k+1]
						pixels_edge_rz = images_rz[i][j][k-1], images_rz[i][j+1][k-1], images_rz[i][j+1][k], images_rz[i][j+1][k+1], images_rz[i][j][k+1]
						pixels_edge_iz = images_iz[i][j][k-1], images_iz[i][j+1][k-1], images_iz[i][j+1][k], images_iz[i][j+1][k+1], images_iz[i][j][k+1]

					if (np.max(images_gr[i][j][k]) > -99):
						if (abs(images_gr[i][j][k]) -abs(np.mean(pixels_edge_gr))) > 0.3:
							images_gr[i][j][k] = np.mean(pixels_edge_gr)
					if (np.max(images_gi[i][j][k]) > -99):
						if (abs(images_gi[i][j][k]) -abs(np.mean(pixels_edge_gi))) > 0.3:
							images_gi[i][j][k] = np.mean(pixels_edge_gi)
					if (np.max(images_gz[i][j][k]) > -99):
						if (abs(images_gz[i][j][k]) -abs(np.mean(pixels_edge_gz))) > 0.3:
							images_gz[i][j][k] = np.mean(pixels_edge_gz)
					if (np.max(images_ri[i][j][k]) > -99):
						if (abs(images_ri[i][j][k]) -abs(np.mean(pixels_edge_ri))) > 0.3:
							images_ri[i][j][k] = np.mean(pixels_edge_ri)
					if (np.max(images_rz[i][j][k]) > -99):
						if (abs(images_rz[i][j][k]) -abs(np.mean(pixels_edge_rz))) > 0.3:
							images_rz[i][j][k] = np.mean(pixels_edge_rz)
					if (np.max(images_iz[i][j][k]) > -99):
						if (abs(images_iz[i][j][k]) -abs(np.mean(pixels_edge_iz))) > 0.3:
							images_iz[i][j][k] = np.mean(pixels_edge_iz)







	for i in tqdm(range(0, len(images_g))):
		for j in range(0,23):
			for k in range(0,23):
				#left side right side
				if j>0 and j<22 and (k<1 or k>21):
					if k<1: #left side

						pixels_edge_gr = images_gr[i][j-1][k], images_gr[i][j-1][k+1], images_gr[i][j][k+1], images_gr[i][j+1][k+1], images_gr[i][j+1][k]
						pixels_edge_gi = images_gi[i][j-1][k], images_gi[i][j-1][k+1], images_gi[i][j][k+1], images_gi[i][j+1][k+1], images_gi[i][j+1][k]
						pixels_edge_gz = images_gz[i][j-1][k], images_gz[i][j-1][k+1], images_gz[i][j][k+1], images_gz[i][j+1][k+1], images_gz[i][j+1][k]
						pixels_edge_ri = images_ri[i][j-1][k], images_ri[i][j-1][k+1], images_ri[i][j][k+1], images_ri[i][j+1][k+1], images_ri[i][j+1][k]
						pixels_edge_rz = images_rz[i][j-1][k], images_rz[i][j-1][k+1], images_rz[i][j][k+1], images_rz[i][j+1][k+1], images_rz[i][j+1][k]
						pixels_edge_iz = images_iz[i][j-1][k], images_iz[i][j-1][k+1], images_iz[i][j][k+1], images_iz[i][j+1][k+1], images_iz[i][j+1][k]
					if k>21: #right side

						pixels_edge_gr = images_gr[i][j-1][k], images_gr[i][j-1][k-1], images_gr[i][j][k-1], images_gr[i][j+1][k-1], images_gr[i][j+1][k]
						pixels_edge_gi = images_gi[i][j-1][k], images_gi[i][j-1][k-1], images_gi[i][j][k-1], images_gi[i][j+1][k-1], images_gi[i][j+1][k]
						pixels_edge_gz = images_gz[i][j-1][k], images_gz[i][j-1][k-1], images_gz[i][j][k-1], images_gz[i][j+1][k-1], images_gz[i][j+1][k]
						pixels_edge_ri = images_ri[i][j-1][k], images_ri[i][j-1][k-1], images_ri[i][j][k-1], images_ri[i][j+1][k-1], images_ri[i][j+1][k]
						pixels_edge_rz = images_rz[i][j-1][k], images_rz[i][j-1][k-1], images_rz[i][j][k-1], images_rz[i][j+1][k-1], images_rz[i][j+1][k]
						pixels_edge_iz = images_iz[i][j-1][k], images_iz[i][j-1][k-1], images_iz[i][j][k-1], images_iz[i][j+1][k-1], images_iz[i][j+1][k]


					if (np.max(images_gr[i][j][k]) > -99):
						if (abs(images_gr[i][j][k]) -abs(np.mean(pixels_edge_gr))) > 0.3:
							images_gr[i][j][k] = np.mean(pixels_edge_gr)
					if (np.max(images_gi[i][j][k]) > -99):
						if (abs(images_gi[i][j][k]) -abs(np.mean(pixels_edge_gi))) > 0.3:
							images_gi[i][j][k] = np.mean(pixels_edge_gi)
					if (np.max(images_gz[i][j][k]) > -99):
						if (abs(images_gz[i][j][k]) -abs(np.mean(pixels_edge_gz))) > 0.3:
							images_gz[i][j][k] = np.mean(pixels_edge_gz)
					if (np.max(images_ri[i][j][k]) > -99):
						if (abs(images_ri[i][j][k]) -abs(np.mean(pixels_edge_ri))) > 0.3:
							images_ri[i][j][k] = np.mean(pixels_edge_ri)
					if (np.max(images_rz[i][j][k]) > -99):
						if (abs(images_rz[i][j][k]) -abs(np.mean(pixels_edge_rz))) > 0.3:
							images_rz[i][j][k] = np.mean(pixels_edge_rz)
					if (np.max(images_iz[i][j][k]) > -99):
						if (abs(images_iz[i][j][k]) -abs(np.mean(pixels_edge_iz))) > 0.3:
							images_iz[i][j][k] = np.mean(pixels_edge_iz)



	for i in tqdm(range(0, len(images_g))):
		for j in range(0,23):
			for k in range(0,23):
				if j>21: #lower corners
					if k<1: #lower left

						pixels_edge_gr = images_gr[i][j][k+1], images_gr[i][j-1][k], images_gr[i][j-1][k+1]
						pixels_edge_gi = images_gi[i][j][k+1], images_gi[i][j-1][k], images_gi[i][j-1][k+1]
						pixels_edge_gz = images_gz[i][j][k+1], images_gz[i][j-1][k], images_gz[i][j-1][k+1]
						pixels_edge_ri = images_ri[i][j][k+1], images_ri[i][j-1][k], images_ri[i][j-1][k+1]
						pixels_edge_rz = images_rz[i][j][k+1], images_rz[i][j-1][k], images_rz[i][j-1][k+1]
						pixels_edge_iz = images_iz[i][j][k+1], images_iz[i][j-1][k], images_iz[i][j-1][k+1]
					if k>21: #lower right

						pixels_edge_gr = images_gr[i][j][k-1], images_gr[i][j-1][k], images_gr[i][j-1][k-1]
						pixels_edge_gi = images_gi[i][j][k-1], images_gi[i][j-1][k], images_gi[i][j-1][k-1]
						pixels_edge_gz = images_gz[i][j][k-1], images_gz[i][j-1][k], images_gz[i][j-1][k-1]
						pixels_edge_ri = images_ri[i][j][k-1], images_ri[i][j-1][k], images_ri[i][j-1][k-1]
						pixels_edge_rz = images_rz[i][j][k-1], images_rz[i][j-1][k], images_rz[i][j-1][k-1]
						pixels_edge_iz = images_iz[i][j][k-1], images_iz[i][j-1][k], images_iz[i][j-1][k-1]

					if k>0 and k<22: #lower side

						pixels_edge_gr = images_gr[i][j][k-1], images_gr[i][j-1][k-1], images_gr[i][j-1][k], images_gr[i][j-1][k+1], images_gr[i][j][k+1]
						pixels_edge_gi = images_gi[i][j][k-1], images_gi[i][j-1][k-1], images_gi[i][j-1][k], images_gi[i][j-1][k+1], images_gi[i][j][k+1]
						pixels_edge_gz = images_gz[i][j][k-1], images_gz[i][j-1][k-1], images_gz[i][j-1][k], images_gz[i][j-1][k+1], images_gz[i][j][k+1]
						pixels_edge_ri = images_ri[i][j][k-1], images_ri[i][j-1][k-1], images_ri[i][j-1][k], images_ri[i][j-1][k+1], images_ri[i][j][k+1]
						pixels_edge_rz = images_rz[i][j][k-1], images_rz[i][j-1][k-1], images_rz[i][j-1][k], images_rz[i][j-1][k+1], images_rz[i][j][k+1]
						pixels_edge_iz = images_iz[i][j][k-1], images_iz[i][j-1][k-1], images_iz[i][j-1][k], images_iz[i][j-1][k+1], images_iz[i][j][k+1]

					if (np.max(images_gr[i][j][k]) > -99):
						if (abs(images_gr[i][j][k]) -abs(np.mean(pixels_edge_gr))) > 0.3:
							images_gr[i][j][k] = np.mean(pixels_edge_gr)
					if (np.max(images_gi[i][j][k]) > -99):
						if (abs(images_gi[i][j][k]) -abs(np.mean(pixels_edge_gi))) > 0.3:
							images_gi[i][j][k] = np.mean(pixels_edge_gi)
					if (np.max(images_gz[i][j][k]) > -99):
						if (abs(images_gz[i][j][k]) -abs(np.mean(pixels_edge_gz))) > 0.3:
							images_gz[i][j][k] = np.mean(pixels_edge_gz)
					if (np.max(images_ri[i][j][k]) > -99):
						if (abs(images_ri[i][j][k]) -abs(np.mean(pixels_edge_ri))) > 0.3:
							images_ri[i][j][k] = np.mean(pixels_edge_ri)
					if (np.max(images_rz[i][j][k]) > -99):
						if (abs(images_rz[i][j][k]) -abs(np.mean(pixels_edge_rz))) > 0.3:
							images_rz[i][j][k] = np.mean(pixels_edge_rz)
					if (np.max(images_iz[i][j][k]) > -99):
						if (abs(images_iz[i][j][k]) -abs(np.mean(pixels_edge_iz))) > 0.3:
							images_iz[i][j][k] = np.mean(pixels_edge_iz)




	clean_griz_cols = np.stack((images_gr, images_gi, images_gz, images_ri, images_rz, images_iz), axis=0)
	np.save('/home/wroster/learning-photoz/PICZL_OZ/run_PICZL/files/FLASH_30/clean_griz_cols.npy', clean_griz_cols)


##################################################################



