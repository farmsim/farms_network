.. FARMS_NETWORK documentation master file, created by
   sphinx-quickstart on Thu Nov 21 19:42:22 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FARMS_NETWORK's documentation!
===========================================

.. warning:: Farmers are currently busy! Documentation is work in progress!!

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials
   modules
   tests
   contributing


Installation steps
-------------------

Requirements
^^^^^^^^^^^^

"Code is only currently verified with Python3"

The installation requires Cython. To install Cython,

.. code-block:: console

    $ pip install cython

(**NOTE** : *Depending on your system installation you may want to use pip3 instead of pip to use python3*)

Installation
^^^^^^^^^^^^^

- Navigate to the root of the directory:

  .. code-block:: console

     $ cd farms_network

- Install system wide with pip:

  .. code-block:: console

     $ pip install .

- Install for user with pip:

  .. code-block:: console

     $ pip install . --user

- Install in developer mode so that you don't have to install every time you make changes to the repository:

  .. code-block:: console

     $ pip install -e .

    - (You may use `--user` option with the above command if you want install only for a user):

- To only compile the module:

  .. code-block:: console

     $ python setup.py build_ext -i


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
