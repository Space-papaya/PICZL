"""
Utility to retrieve paths to demo data packaged within the piczl module.
"""

import importlib.resources as pkg_resources
import os
import sys

# Add the src directory to sys.path to enable module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def get_demo_data_path(exec: str):
    """
    Get the paths to the demo catalog and demo images directory
    for a given execution configuration.

    Parameters
    ----------
    exec : str
        Name of the subdirectory under piczl.demo_data where demo_catalog.fits
        and demo_images/ are located.

    Returns
    -------
    tuple of str
        A tuple containing:
        - Full path to demo_catalog.fits
        - Full path to demo_images/ directory
    """
    base = pkg_resources.files("piczl.demo_data")
    catalog_path = base.joinpath(f"{exec}/demo_catalog.fits")
    images_path = base.joinpath(f"{exec}/demo_images")

    return str(catalog_path), str(images_path)
