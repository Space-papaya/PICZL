import os
import importlib

# Current directory
package_dir = os.path.dirname(__file__)

# Automatically import all .py files in this directory (except __init__.py)
__all__ = []

for filename in os.listdir(package_dir):
	if filename.endswith(".py") and filename != "__init__.py":
		module_name = filename[:-3]
		module = importlib.import_module(f".{module_name}", package=__name__)
		globals()[module_name] = module
		__all__.append(module_name)
