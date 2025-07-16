# Configuration file for the Sphinx documentation builder.
#

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PICZL'
copyright = '2025, William Roster'
author = 'William Roster'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Optional, for Google or NumPy docstrings
    'sphinx.ext.viewcode',  # Optional, adds links to source code
    'sphinx_book_theme',
    'sphinx.ext.autosummary',
    'sphinx_copybutton',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_autodoc_typehints',
    'myst_nb'
]

templates_path = ['_templates']
exclude_patterns = []

source_suffix = [".rst", ".ipynb", ".md"]
pygments_style = 'colorful'
autodoc_default_flags = ["members"]
autosummary_generate = True
napolean_use_rtype = False



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme' #sphinx_rtd_theme

html_theme_options = {
    "repository_url": "https://github.com/williamroster/PICZL",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "logo": {
    "image_light": "_static/PICZL_logo.png",
    "image_dark": "_static/PICZL_logo.png",
}
}
html_logo = "_static/PICZL_logo.png"

html_static_path = ['_static']
