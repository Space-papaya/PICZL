Getting Started
================

1. Create a clean conda environment with fixed dependencies
-----------------------------------------------------------

You want to set up all core dependencies before installing your package. We strongly recommend using a conda 
environment to control Python version compliance and isolate your installation. To prepare the environment upfront:

.. code-block:: bash

    conda create -n piczl_env python=3.10 tensorflow=2.12 tensorflow-probability=0.18 pandas=1.5.3 -c conda-forge
    conda activate piczl_env


2. Install piczl package
------------------------

``piczl`` is distributed with the Python Package Index `(PyPI) <https://pypi.org/project/PICZL/>`_, and 
thus the simplest way to install it is with pip:

.. code-block:: bash

    pip install piczl

.. warning:: 
    If you run ``pip install piczl`` in a bare environment first, the ``install_requires`` in ``pyproject.toml`` will pull in specific versions of dependencies, 
    which might conflict or upgrade/downgrade your packages unexpectedly. 

We have prepared an introductory notebooks. In order to run it you must install jupyterlab with the following commands:

.. code-block:: bash

    conda install -c conda-forge jupyterlab
    # and create a kernel which has access to this environment
    python -m ipykernel install --user --name <kernel_name>

At this stage, the following Python snippet should work:

.. code-block:: python

    import piczl as pc

In order to run ``piczl`` we need auxiliary data in the form of a catalog hosting LS10 columns for a set of galaxies/AGN and their subsequent imaging cutouts. A small demo set of both is shipped with the code (~0.3Gb). The import and handling of samples is explained in more detail below. We can now test predicting photometric redshifts for the demo data via:

.. code-block:: python

     import ``piczl`` as pc
     pc.execute.run.predict_redshifts()


The following Python snippet is the most basic example to test the installation has worked. 
This will generate a single output file stored in the ``demo/result/`` folder. The user may define storage locations later.
You can also get an example notebook running this code `here <https://github.com>`_.
