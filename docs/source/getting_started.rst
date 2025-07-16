Getting Started
---------------



Installation
============
``PICZL`` is distributed with the Python Package Index `(PyPI) <https://pypi.org/project/PICZL/>`_, and 
thus the simplest way to install it is with pip:

.. code-block:: bash

    pip install PICZL

    
We also reccommend using a conda 
environment to control Python version and isolate your installation:

.. code-block:: bash
    
    #We strongly recommend using Python 3.10 to comply with TensorFlow's requirements
    conda create -n <env_name> python=3.10 tensorflow=2.12 tensorflow-probability=0.18
    conda activate <env_name>
    pip install PICZL
    # We have prepared an introductory notebooks. In order to run it
    # you must install jupyterlab with the following commands.
    conda install -c conda-forge jupyterlab
    # And create a kernel which has access to this environment
    python -m ipykernel install --user --name <kernel_name>

A this stage, the following Python snippet should work:

.. code-block:: python

    import PICZL as pc

    
.. warning:: 
    Make sure that the code dependencies are in place in your env. This is currently the biggest breakpoint.

In order to run PICZL we need auxiliary data in the form of a catalog hosting LS10 columns and the subsequent imaging cutouts. A small demo set of both is shipped with the code (~0.3Gb). These are explained
in more detail below. We can therefore test predicting photometric redshifts for the demo data via:

.. code-block:: python

     import PICZL as pc
     pc.run_predictions('Add paths')


The following Python snippet is the most basic example to test the installation has worked. 
This will generate a single output file stored in the demo folder. The user may define storage locations later.
You can also get an example notebook running this code `here <https://github.com>`_.
