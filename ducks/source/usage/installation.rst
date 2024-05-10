Installation steps
-------------------

Requirements
^^^^^^^^^^^^

"Code is only currently tested with Python3"

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
