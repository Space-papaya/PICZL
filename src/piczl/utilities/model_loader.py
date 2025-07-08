import importlib.resources as pkg_resources
from pathlib import Path

def get_model_path(mode, model_filename):
    """
    Return the full path to a model file inside the piczl package.

    Parameters:
        mode (str): 'active' or 'inactive' to select model subfolder.
        model_filename (str): filename of the model file.

    Returns:
        Path: a pathlib.Path object pointing to the model file.
    """
    package_name = f"piczl.models.{mode}"

    with pkg_resources.path(package_name, model_filename) as model_path:
        return model_path
