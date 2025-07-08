import os
import pkgutil
import importlib

# Get the directory of this __init__.py file
__all__ = []

package_dir = os.path.dirname(__file__)

# Iterate through all modules and subpackages in this package
for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
    if not module_name.startswith("_"):
        module = importlib.import_module(f".{module_name}", package=__name__)
        globals()[module_name] = module
        __all__.append(module_name)
