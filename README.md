# PICZL

**PICZL** (Photometrically Infered CNN Redshift Likelihoods) is a modular Python package for estimating photometric redshifts 
of galaxies and AGN as observed in the 10th data release of the DESI Legacy survey by using both catalog-based photometry and imaging data. 
It supports inference using ensemble deep learning models and includes tools for: 

- Gaussian Mixture Model (GMM) PDF outputs with ensemble PDFs.
- Classification of redshift PDF degeneracies.
- Demonstration data and inference script for immediate use after installation.
- Retraining the algorithm fine-tuned to a target object population.

PICZL is designed for easy deployment, benchmarking, and use with custom data, for quick testing after installation.


## Installation

PICZL is available as a Python package and can be installed via `pip`. It requires **Python == 3.10** and a working environment with TensorFlow 
with optional GPU acceleration. Ensure you have installed the appropriate CUDA toolkit and cuDNN libraries. For most systems, 
installing the TensorFlow package will auto-detect GPU support if available.


### 1. Install via pip

```Shell
conda create -n *env-name* python=3.10
conda activate *env-name*
pip install piczl
```
