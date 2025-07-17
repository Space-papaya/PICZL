Getting Started
================

Create conda environment with fixed dependencies
-----------------------------------------------------------

You want to set up all core dependencies before installing your package. We strongly recommend using a conda 
environment to control Python version compliance and isolate your installation. To prepare the environment upfront:

.. code-block:: bash

    conda create -n piczl_env python=3.10 tensorflow=2.12 tensorflow-probability=0.18 pandas=1.5.3 -c conda-forge
    conda activate piczl_env


Install piczl package
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

In order to run ``piczl`` we need auxiliary data in the form of a catalog hosting LS10 columns for a set of galaxies/AGN and their subsequent imaging cutouts. A small demo set of both is shipped with the code (~20Mb). The import and handling of samples is explained in more detail below. We can now test predicting photometric redshifts for the demo data via:

.. code-block:: python

     import piczl as pc
     output = pc.execute.run.predict_redshifts()

This minimal Python example verifies that the installation was successful. It produces a single output file containing a demo catalog with additional redshift-related columns in a DataFrame.
You can customize the output location later as needed. A full example notebook running this code is available `here <https://github.com>`_.


The Configuration Keywords
==========================

Most users will want to apply ``piczl`` to their own datasets, which requires adjusting the input data path and selecting the appropriate inference mode. By default, the package includes pre-trained models tailored for both ``inactive`` and ``active`` galaxy populations.

.. code-block:: python 

    predict_redshifts(data_path="/../..",
        image_path ="/../..",
        subsample=False,
        catalog_name="..",
        use_demo_data=False)


For more advanced use cases, users can choose to retrain or fine-tune these models on a specific target population. This can be done either by keeping the default setup and switching the mode, or by exploring and customizing the training configuration, model architectures, and other adjustable components. 

