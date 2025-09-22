==============
 Installation
==============

Code is currently tested with Python3 version ["3.11", "3.13"] on ubuntu-latest, macos-latest, windows-latest using docker images on GitHub.
We recommend using a virtual environment to avoid conflicts with other Python packages.
This guide will help you install farms_network and its dependencies.

Prerequisites
------------

Before installing farms_network, ensure you have the following prerequisites:

* Python 3.11 or higher
* pip (Python package installer)
* A C++ compiler (for building extensions)


Basic Installation
----------------

You can install farms_network using pip:

.. code-block:: bash

      pip git+install https://github.com/farmsim/farms_network.git

(You may use `--user` option with the above command if you want install only for a user)

(**NOTE** : *Depending on your system installation you may want to use pip3 instead of pip to use python3*)

Development Installation
----------------------

For development purposes, you can install farms_network from source:

.. code-block:: bash

   git clone https://github.com/username/farms_network.git
   cd farms_network
   pip install -e .[dev]

The `[dev]` flag will install additional dependencies needed for development.

Optional Dependencies
-------------------

farms_network has several optional dependencies for extended functionality:

..
   * **GPU Support**: For GPU acceleration

     .. code-block:: bash

        pip install farms_network[gpu]

* **Visualization**: For advanced visualization features

  .. code-block:: bash

     pip install farms_network[viz]


Updating farms_network
--------------------

To update an existing installation to the latest version:

.. code-block:: bash

   pip install --upgrade farms_network

To update to a specific version:

.. code-block:: bash

   pip install --upgrade farms_network==<version>

Note: After updating, it's recommended to restart any running applications or kernels using farms_network.

Troubleshooting
--------------

Common Installation Issues
^^^^^^^^^^^^^^^^^^^^^^^^

1. **Compiler errors**: Ensure you have a compatible C++ compiler installed
2. **Missing dependencies**: Try installing the package with all optional dependencies:

   .. code-block:: bash

      pip install farms_network[all]

3. **Version conflicts**: If you encounter dependency conflicts, try creating a fresh virtual environment:

   .. code-block:: bash

      python -m venv farms_env
      source farms_env/bin/activate  # On Unix
      # or
      farms_env\Scripts\activate  # On Windows
