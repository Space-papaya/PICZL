
from astropy.table import Table
import numpy as np
import pandas as pd


def append_output(dataset, pwd, catalog_name, ens_modes, l1s, u1s, degeneracy):
    """
    Append redshift estimation results and metadata to the dataset and save as a FITS file.

    This function takes a pandas DataFrame `dataset` and appends columns for:
        - ensemble mode redshift estimates,
        - lower 1-sigma confidence intervals,
        - upper 1-sigma confidence intervals,
        - PDF degeneracy classification.

    Then it converts the DataFrame to an Astropy Table and writes it out as a FITS file.

    Parameters:
        dataset (pd.DataFrame): Input catalog dataframe to be augmented.
        pwd (str): Directory path where the FITS file will be saved.
        catalog_name (str): Base filename (prefix) for the output FITS file.
        ens_modes (array-like): Ensemble mode redshift estimates to add.
        l1s (array-like): Lower 1-sigma confidence intervals to add.
        u1s (array-like): Upper 1-sigma confidence intervals to add.
        degeneracy (array-like): PDF degeneracy classification to add.

    Returns:
        None
    """
    dataset['phz_inact'] = ens_modes
    dataset['lower_1sig_inact'] = l1s
    dataset['upper_1sig_inact'] = u1s
    dataset['pdf_degenercy'] = degeneracy

    tab = Table.from_pandas(dataset)
    tab.write(pwd + catalog_name + 'demo_catalog.fits', overwrite=True)
