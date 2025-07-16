.. PICZL documentation master file, created by
   sphinx-quickstart on Tue Jul  8 16:50:21 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PICZL's documentation!
========================================================================================

``PICZL`` is a modular Python package for estimating photometric redshifts of galaxies and AGN as observed in the 10th data release of the `DESI Legacy Survey <https://www.legacysurvey.org>`_ by using both catalog-based photometry and imaging data. It supports inference using ensemble deep learning models and includes tools for: 

- Gaussian Mixture Model (GMM) outputs with ensemble PDFs.
- Classification of redshift PDF degeneracies.
- Demonstration data for immediate use after installation.
- Retraining the algorithm fine-tuned to a target object population.

PICZL is designed for easy deployment, benchmarking, and use with custom data. If you find this work useful, please cite::

    @ARTICLE{2024A&A...692A.260R,
         author = {{Roster}, W. and {Salvato}, M. and {Krippendorf}, S. and {Saxena}, A. and {Shirley}, R. and {Buchner}, J. and {Wolf}, J. and {Dwelly}, T. and {Bauer}, F.~E. and {Aird}, J. and {Ricci}, C. and {Assef}, R.~J. and {Anderson}, S.~F. and {Liu}, X. and {Merloni}, A. and {Weller}, J. and {Nandra}, K.},
         title = "{PICZL: Image-based photometric redshifts for AGN}",
         journal = {\aap},
         keywords = {methods: statistical, techniques: photometric, galaxies: active, quasars: supermassive black holes, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics, Statistics - Machine Learning},
         year = 2024,
         month = dec,
         volume = {692},
         eid = {A260},
         pages = {A260},
         doi = {10.1051/0004-6361/202452361},
         archivePrefix = {arXiv},
         eprint = {2411.07305},
         primaryClass = {astro-ph.GA},
         adsurl = {https://ui.adsabs.harvard.edu/abs/2024A&A...692A.260R},
         adsnote = {Provided by the SAO/NASA Astrophysics Data System}
         }

'''
Table of contents
-----------------

.. toctree::
    :maxdepth: 1
    :caption: Home page <self>

    intro/Getting Started <getting_started>

.. toctree::
    :hidden:
    PICZL

