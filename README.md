# PICZL [![build](https://github.com/williamroster/PICZL/actions/workflows/codecov.yml/badge.svg)](https://github.com/williamroster/PICZL/actions/workflows/codecov.yml)

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


### Install via pip

```Shell
conda create -n *env-name* python=3.10
conda activate *env-name*
pip install piczl
```

### Verify installation

You can test the package and its dependencies by running a prediction on demo data:

```python
from piczl.execute.run import predict_redshifts

predict_redshifts(catalog_name="demo_catalog.fits", mode="inactive", subsample=True)
```

**Parameters:**

- DATA_PATH: Directory containing the catalog and image data
- catalog_name: Filename of the FITS catalog
- mode: "active" for AGN models or "inactive" for galaxy models
- subsample: If True, runs on a small number of sources for quick testing

**Example Output:**

- z_modes: The peak of each redshift PDF
- l1s, u1s: Upper and lower 1 sigma errors from 68% highest probability density (HPD) interval bounds
- degeneracy: Classification of PDF shape (none, light, medium, strong)

These predictions are saved in a FITS file called:
```PICZL_predictions_<original_catalog_name>.fits```



## Contributing and Support

We welcome contributions from the community to improve PICZL! If you'd like to contribute:

- **Fork** the repository and create a feature branch.
- Add **tests** for any new features or bug fixes.
- Submit a **pull request** with a clear description of your changes.

### Reporting Issues

If you encounter bugs or have feature requests, please open an issue on the GitHub repository with:

- A clear description of the problem or suggestion.
- Steps to reproduce the issue (if applicable).
- Your environment details (Python version, OS, GPU availability).

### Getting Help

For questions or usage help, you can:

- Check the **README** and code documentation.
- Reach out through the repository’s **discussion** board or contact the maintainers.


## License and Citation

PICZL is released under the [MIT License], allowing free use, modification, and distribution.

If you use PICZL in your research or projects, please cite the following paper:

**PICZL: Image-based photometric redshifts for AGN**
*Authors: William Roster et al.*, *A&A*, *2024*, DOI: https://doi.org/10.1051/0004-6361/202452361

Proper citation helps support continued development and maintenance of PICZL.

---

Thank you for using PICZL!
