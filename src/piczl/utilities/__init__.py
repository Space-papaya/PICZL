
"""
Dynamically import all Python modules in this package directory (excluding __init__.py).

This enables importing all submodules automatically when the package is imported,
and populates the __all__ list for controlled exports.
"""


import os
import importlib

package_dir = os.path.dirname(__file__)
__all__ = []

for filename in os.listdir(package_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]
        module = importlib.import_module(f".{module_name}", package=__name__)
        globals()[module_name] = module
        __all__.append(module_name)
