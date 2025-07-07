import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from piczl.core.estimator import run_estimation
from piczl.utilities import *


def predict_redshifts(DATA_PATH=None, catalog_name=None, mode='active', subsample=True, use_demo_data=True):
    """
    Run photometric redshift prediction for a given catalog and data path.

    This function calls the core redshift estimator, prints a summary of results,
    and saves the predictions along with metadata.

    Parameters
    ----------
    DATA_PATH : str or None
        Directory path containing catalog and image data.
	Ignored if 'use_demo_data' is used.
    catalog_name : str
        Filename of the catalog to process.
    mode : str
        Processing mode; typically 'active' or 'inactive'.
    subsample : bool, optional
        Whether to run on a subsample of the data for testing (default is True).
    use_demo_data : bool
        If True, used packaged demo data instead of user paths.

    Returns
    -------
    None
        Prints output to console and saves results to disk.
    """


    if use_demo_data:
        catalog_path, image_path = load_demo_data.get_demo_data_path('run')
    else:
        catalog_path = os.path.join(DATA_PATH, catalog_name)
        image_data = DATA_PATH

    z_modes, l1s, u1s, degeneracy, dataset = run_estimation(
        catalog_path=catalog_path,
        image_path=image_path,
        mode=mode,
        sub_sample=subsample,
        max_sources=20
    )

    print('\n')
    print(' >> Output header:')
    print("z_peak:", [float(f"{z:.3f}") for z in z_modes[:5]])
    print("l1s:", l1s[:5])
    print("u1s:", u1s[:5])
    print("degeneracy:", degeneracy[:5])


    file_name = 'PICZL_predictions_'
    save_path = os.path.abspath(os.path.join(catalog_path, '..'))
    output.append_output(dataset, save_path + '/result/', file_name, z_modes, l1s, u1s, degeneracy)

    print(">> Results saved.")
