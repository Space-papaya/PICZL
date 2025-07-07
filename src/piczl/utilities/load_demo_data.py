import importlib.resources as pkg_resources
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def get_demo_data_path(exec):
    """
    Returns the absolute path to the test catalog and images folder inside the package.
    """
    base = pkg_resources.files("piczl.demo_data")
    catalog_path = base.joinpath(f"{exec}/demo_catalog.fits")
    images_path = base.joinpath(f"{exec}/demo_images")

    return str(catalog_path), str(images_path)
