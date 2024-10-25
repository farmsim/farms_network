Options Module Documentation
============================

This module contains the configuration options for neural network models, including options for nodes, edges, integration, and visualization.

NetworkOptions Class
--------------------
.. autoclass:: farms_network.core.options.NetworkOptions
   :members:
   :undoc-members:

   **Attributes:**

   - **directed** (bool): Whether the network is directed. Default is `True`.
   - **multigraph** (bool): Whether the network allows multiple edges between nodes. Default is `False`.
   - **graph** (dict): Graph properties (e.g., name). Default is `{"name": ""}`.
   - **units** (optional): Units for the network. Default is `None`.
   - **integration** (:class:`IntegrationOptions`): Options for numerical integration. Default values shown in the table below.

IntegrationOptions Class
------------------------
.. autoclass:: farms_network.core.options.IntegrationOptions
   :members:
   :undoc-members:

   The default values for `IntegrationOptions` are as follows:

   +------------+-------------------+
   | Parameter  | Default Value     |
   +------------+-------------------+
   | timestep   | ``1e-3``          |
   +------------+-------------------+
   | integrator | ``"dopri5"``      |
   +------------+-------------------+
   | method     | ``"adams"``       |
   +------------+-------------------+
   | atol       | ``1e-12``         |
   +------------+-------------------+
   | rtol       | ``1e-6``          |
   +------------+-------------------+
   | max_step   | ``0.0``           |
   +------------+-------------------+
   | checks     | ``True``          |
   +------------+-------------------+

NodeOptions Class
-----------------
.. autoclass:: farms_network.core.options.NodeOptions
   :members:
   :undoc-members:

   **Attributes:**

   - **name** (str): Name of the node.
   - **model** (str): Node model type.
   - **parameters** (:class:`NodeParameterOptions`): Node-specific parameters.
   - **state** (:class:`NodeStateOptions`): Node state options.

NodeParameterOptions Class
--------------------------
.. autoclass:: farms_network.core.options.NodeParameterOptions
   :members:
   :undoc-members:

   The default values for `NodeParameterOptions` are as follows:

   +----------------+----------------+
   | Parameter      | Default Value  |
   +================+================+
   | c_m            | ``10.0`` pF    |
   +----------------+----------------+
   | g_leak         | ``2.8`` nS     |
   +----------------+----------------+
   | e_leak         | ``-60.0`` mV   |
   +----------------+----------------+
   | v_max          | ``0.0`` mV     |
   +----------------+----------------+
   | v_thr          | ``-50.0`` mV   |
   +----------------+----------------+
   | g_syn_e        | ``10.0`` nS    |
   +----------------+----------------+
   | g_syn_i        | ``10.0`` nS    |
   +----------------+----------------+
   | e_syn_e        | ``-10.0`` mV   |
   +----------------+----------------+
   | e_syn_i        | ``-75.0`` mV   |
   +----------------+----------------+

NodeStateOptions Class
----------------------
.. autoclass:: farms_network.core.options.NodeStateOptions
   :members:
   :undoc-members:

   **Attributes:**

   - **initial** (list of float): Initial state values.
   - **names** (list of str): State variable names.

EdgeOptions Class
-----------------
.. autoclass:: farms_network.core.options.EdgeOptions
   :members:
   :undoc-members:

   **Attributes:**

   - **from_node** (str): Source node of the edge.
   - **to_node** (str): Target node of the edge.
   - **weight** (float): Weight of the edge.
   - **type** (str): Edge type (e.g., excitatory, inhibitory).

EdgeVisualOptions Class
-----------------------
.. autoclass:: farms_network.core.options.EdgeVisualOptions
   :members:
   :undoc-members:

   **Attributes:**

   - **color** (list of float): Color of the edge.
   - **label** (str): Label for the edge.
   - **layer** (str): Layer in which the edge is displayed.

LIDannerParameterOptions Class
------------------------------
.. autoclass:: farms_network.core.options.LIDannerParameterOptions
   :members:
   :undoc-members:

   The default values for `LIDannerParameterOptions` are as follows:

   +----------------+----------------+
   | Parameter      | Default Value  |
   +================+================+
   | c_m            | ``10.0`` pF    |
   +----------------+----------------+
   | g_leak         | ``2.8`` nS     |
   +----------------+----------------+
   | e_leak         | ``-60.0`` mV   |
   +----------------+----------------+
   | v_max          | ``0.0`` mV     |
   +----------------+----------------+
   | v_thr          | ``-50.0`` mV   |
   +----------------+----------------+
   | g_syn_e        | ``10.0`` nS    |
   +----------------+----------------+
   | g_syn_i        | ``10.0`` nS    |
   +----------------+----------------+
   | e_syn_e        | ``-10.0`` mV   |
   +----------------+----------------+
   | e_syn_i        | ``-75.0`` mV   |
   +----------------+----------------+

LIDannerNaPParameterOptions Class
---------------------------------
.. autoclass:: farms_network.core.options.LIDannerNaPParameterOptions
   :members:
   :undoc-members:

   The default values for `LIDannerNaPParameterOptions` are as follows:

   +----------------+----------------+
   | Parameter      | Default Value  |
   +================+================+
   | c_m            | ``10.0`` pF    |
   +----------------+----------------+
   | g_nap          | ``4.5`` nS     |
   +----------------+----------------+
   | e_na           | ``50.0`` mV    |
   +----------------+----------------+
   | v1_2_m         | ``-40.0`` mV   |
   +----------------+----------------+
   | k_m            | ``-6.0``       |
   +----------------+----------------+
   | v1_2_h         | ``-45.0`` mV   |
   +----------------+----------------+
   | k_h            | ``4.0``        |
   +----------------+----------------+
   | v1_2_t         | ``-35.0`` mV   |
   +----------------+----------------+
   | k_t            | ``15.0``       |
   +----------------+----------------+
   | g_leak         | ``4.5`` nS     |
   +----------------+----------------+
   | e_leak         | ``-62.5`` mV   |
   +----------------+----------------+
   | tau_0          | ``80.0`` ms    |
   +----------------+----------------+
   | tau_max        | ``160.0`` ms   |
   +----------------+----------------+
   | v_max          | ``0.0`` mV     |
   +----------------+----------------+
   | v_thr          | ``-50.0`` mV   |
   +----------------+----------------+
   | g_syn_e        | ``10.0`` nS    |
   +----------------+----------------+
   | g_syn_i        | ``10.0`` nS    |
   +----------------+----------------+
   | e_syn_e        | ``-10.0`` mV   |
   +----------------+----------------+
   | e_syn_i        | ``-75.0`` mV   |
   +----------------+----------------+
