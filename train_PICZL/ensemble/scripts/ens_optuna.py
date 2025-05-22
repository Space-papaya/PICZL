
import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.table import Table
import optuna

########################################################################################################

test_pdfs = np.load('/home/wroster/learning-photoz/PICZL_OZ/train_PICZL/ensemble/files/optuna_test_pdfs.npy')
all_pdfs_sorted = test_pdfs #all_pdf

val_t = Table.read('/home/wroster/learning-photoz/PICZL_OZ/train_PICZL/output/test_sample.fits')
vals = val_t.to_pandas()
labels_val_crps = np.array(vals['Z'])

# Create the array using a list comprehension
num_objects = len(labels_val_crps)
num_samples = 4001
samples = np.array([np.linspace(0.002, 8, num_samples) for _ in range(num_objects)])

#print(all_pdfs_sorted.shape)
#print(labels_val_crps.shape)


########################################################################################################


# Define the objective function to minimize
def objective_function(weights, all_pdfs_sorted, samples, labels_val_crps):
    # Compute the weighted sum of pdf scores
    sum_pdf_scores = np.sum([weights[i] * all_pdfs_sorted[i] for i in range(len(weights))], axis=0)
    sum_modes = samples[1][np.argmax(sum_pdf_scores, axis=1)]

    # Compute bias_z and frac_z
    bias_z = np.abs(labels_val_crps - sum_modes)
    frac_z = bias_z / (1 + labels_val_crps)

    # Calculate the fraction of outliers
    out_frac = np.sum(frac_z > 0.15) / len(labels_val_crps)

    return out_frac


# Custom callback to control verbosity based on improvement
def custom_callback(study, trial):
    if study.best_trial is None or trial.number == study.best_trial.number:
        # First trial or new best trial found
        print(f"\n New best trial found! Trial number: {trial.number}, Value: {trial.value} \n")

best_optuna_values = []

# Custom callback function to track best values
def track_best_value(study, trial):
    # Store the best value after each trial
    best_optuna_values.append(study.best_value)


# Define the Optuna objective function
def objective(trial):
    # Generate random weights within the trial
    weights = [trial.suggest_uniform(f'weight_{i}', 0, 1) for i in range(n_weights)]

    # Evaluate the objective function with the normalized weights
    out_frac = objective_function(weights, all_pdfs_sorted, samples, labels_val_crps)

    return out_frac



n_trials = 5000

# Number of PDFs (weights)
n_weights = len(all_pdfs_sorted)

# Set up Optuna study and optimize the weights
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=n_trials, callbacks=[custom_callback,track_best_value])

# Retrieve the best weights based on the minimum out_frac value
best_weights = [study.best_params[f'weight_{i}'] for i in range(n_weights)]
min_out_frac = study.best_value


# Normalize the best weights to sum up to 1
best_weights_sum = np.sum(best_weights)
normalized_best_weights = [w / best_weights_sum for w in best_weights]


print("Best normalized best weights:", normalized_best_weights)
print("Minimum out_frac value:", min_out_frac)

########################################################################################################


x = np.linspace(1,n_trials,n_trials)
plt.plot(x, best_optuna_values)
plt.ylabel('Outlier fraction')
plt.xlabel('Iterations')
plt.savefig('/home/wroster/learning-photoz/PICZL_OZ/train_PICZL/ensemble/files/progress_'+str(n_trials)+'_combined.pdf')
plt.clf()


# Define the output file path
output_file_path = '/home/wroster/learning-photoz/PICZL_OZ/train_PICZL/ensemble/files/weights_'+str(n_trials)+'_combined.txt'

# Open the file in write mode and write the content
with open(output_file_path, 'w') as f:
    f.write("\n Number of iterations: {}\n".format(n_trials))
    f.write("\n Best normalized weights: {}\n".format(normalized_best_weights))
    f.write("\n Minimum out_frac value: {}\n".format(min_out_frac))

print("Output saved to:", output_file_path)


