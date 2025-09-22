Configuration Options
====================

This module provides configuration options for neural network components and simulations.


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
..

   Network Options
   -------------

   .. autoclass:: farms_network.options.NetworkOptions
      :members:
      :undoc-members:

      The main configuration class for neural networks. Controls network structure, simulation parameters, and logging.

      **Key Attributes:**

      - ``directed``: Network directionality (default: True)
      - ``multigraph``: Allow multiple edges between nodes (default: False)
      - ``nodes``: List of :class:`NodeOptions`
      - ``edges``: List of :class:`EdgeOptions`
      - ``integration``: :class:`IntegrationOptions` for simulation settings
      - ``logs``: :class:`NetworkLogOptions` for data collection
      - ``random_seed``: Seed for reproducibility

   Node Options
   -----------

   .. autoclass:: farms_network.options.NodeOptions
      :members:
      :undoc-members:

      Base class for neuron configuration.

      **Key Attributes:**

      - ``name``: Unique identifier
      - ``model``: Neural model type
      - ``parameters``: Model-specific parameters
      - ``state``: Initial state variables
      - ``visual``: Visualization settings
      - ``noise``: Noise configuration

   Available Node Models
   ^^^^^^^^^^^^^^^^^^^

   * :class:`RelayNodeOptions`: Simple signal relay
   * :class:`LinearNodeOptions`: Linear transformation
   * :class:`ReLUNodeOptions`: Rectified linear unit
   * :class:`OscillatorNodeOptions`: Phase-amplitude oscillator
   * :class:`LIDannerNodeOptions`: Leaky integrator (Danner model)
   * :class:`LINaPDannerNodeOptions`: Leaky integrator with NaP

   Edge Options
   -----------

   .. autoclass:: farms_network.options.EdgeOptions
      :members:
      :undoc-members:

      Configuration for synaptic connections.

      **Key Attributes:**

      - ``source``: Source node name
      - ``target``: Target node name
      - ``weight``: Connection strength
      - ``type``: Synapse type (e.g., excitatory/inhibitory)
      - ``parameters``: Model-specific parameters
      - ``visual``: Visualization settings

   Integration Options
   -----------------

   .. autoclass:: farms_network.options.IntegrationOptions
      :members:
      :undoc-members:

      Numerical integration settings.

      **Key Attributes:**

      - ``timestep``: Integration step size
      - ``n_iterations``: Number of iterations
      - ``integrator``: Integration method (e.g., 'rk4')
      - ``method``: Solver method
      - ``atol``: Absolute tolerance
      - ``rtol``: Relative tolerance

   Example Usage
   -----------

   Basic network configuration:

   .. code-block:: python

       from farms_network.options import NetworkOptions, LIDannerNodeOptions, EdgeOptions

       # Create network options
       net_opts = NetworkOptions(
           directed=True,
           integration=IntegrationOptions.defaults(timestep=0.1),
           logs=NetworkLogOptions(n_iterations=1000)
       )

       # Add nodes
       node1 = LIDannerNodeOptions(
           name="neuron1",
           parameters=LIDannerNodeParameterOptions.defaults()
       )
       net_opts.add_node(node1)

       # Add edges
       edge = EdgeOptions(
           source="neuron1",
           target="neuron2",
           weight=0.5,
           type="excitatory"
       )
       net_opts.add_edge(edge)


   Configuration Options
   ====================

   Base Options
   -----------

   Node Options
   ^^^^^^^^^^^

   .. autoclass:: farms_network.options.NodeOptions
      :members:

   Base class for all node configurations.

   **Attributes:**

   - ``name`` (str): Unique identifier
   - ``model`` (str): Neural model type
   - ``parameters`` (NodeParameterOptions): Model-specific parameters
   - ``state`` (NodeStateOptions): Initial state variables
   - ``visual`` (NodeVisualOptions): Visualization settings
   - ``noise`` (NoiseOptions): Noise configuration

   Node Visual Options
   ^^^^^^^^^^^^^^^^^

   .. autoclass:: farms_network.options.NodeVisualOptions
      :members:

   Visualization settings for nodes.

   **Default Values:**

   .. list-table::
      :header-rows: 1

      * - Parameter
        - Default Value
        - Description
      * - position
        - [0.0, 0.0, 0.0]
        - 3D coordinates
      * - radius
        - 1.0
        - Node size
      * - color
        - [1.0, 0.0, 0.0]
        - RGB values
      * - label
        - "n"
        - Display label
      * - layer
        - "background"
        - Rendering layer
      * - latex
        - "{}"
        - LaTeX formatting

   Node State Options
   ^^^^^^^^^^^^^^^^^

   .. autoclass:: farms_network.options.NodeStateOptions
      :members:

   Base class for node states.

   **Attributes:**

   - ``initial`` (List[float]): Initial state values
   - ``names`` (List[str]): State variable names

   Node Models
   -----------

   Relay Node
   ^^^^^^^^^

   .. autoclass:: farms_network.options.RelayNodeOptions
      :members:

   Simple signal relay node.

   Linear Node
   ^^^^^^^^^^

   .. autoclass:: farms_network.options.LinearNodeOptions
      :members:

   Linear transformation node.

   **Default Parameters:**

   .. list-table::
      :header-rows: 1

      * - Parameter
        - Default Value
        - Description
      * - slope
        - 1.0
        - Linear transformation slope
      * - bias
        - 0.0
        - Constant offset

   ReLU Node
   ^^^^^^^^

   .. autoclass:: farms_network.options.ReLUNodeOptions
      :members:

   Rectified Linear Unit node.

   **Default Parameters:**

   .. list-table::
      :header-rows: 1

      * - Parameter
        - Default Value
        - Description
      * - gain
        - 1.0
        - Amplification factor
      * - sign
        - 1
        - Direction (+1/-1)
      * - offset
        - 0.0
        - Activation threshold

   Oscillator Node
   ^^^^^^^^^^^^^

   .. autoclass:: farms_network.options.OscillatorNodeOptions
      :members:

   Phase-amplitude oscillator.

   **Default Parameters:**

   .. list-table::
      :header-rows: 1

      * - Parameter
        - Default Value
        - Description
      * - intrinsic_frequency
        - 1.0
        - Base frequency (Hz)
      * - nominal_amplitude
        - 1.0
        - Base amplitude
      * - amplitude_rate
        - 1.0
        - Amplitude change rate

   Leaky Integrator Node (Danner)
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   .. autoclass:: farms_network.options.LIDannerNodeOptions
      :members:

   Leaky integrator with Danner dynamics.

   **Default Parameters:**

   .. list-table::
      :header-rows: 1

      * - Parameter
        - Default Value
        - Description
      * - c_m
        - 10.0
        - Membrane capacitance (pF)
      * - g_leak
        - 2.8
        - Leak conductance (nS)
      * - e_leak
        - -60.0
        - Leak reversal potential (mV)
      * - v_max
        - 0.0
        - Maximum voltage (mV)
      * - v_thr
        - -50.0
        - Threshold voltage (mV)
      * - g_syn_e
        - 10.0
        - Excitatory synaptic conductance (nS)
      * - g_syn_i
        - 10.0
        - Inhibitory synaptic conductance (nS)
      * - e_syn_e
        - -10.0
        - Excitatory synaptic reversal potential (mV)
      * - e_syn_i
        - -75.0
        - Inhibitory synaptic reversal potential (mV)
      * - tau_ch
        - 5.0
        - Cholinergic time constant (ms)

   LINaP Node (Danner)
   ^^^^^^^^^^^^^^^^^

   .. autoclass:: farms_network.options.LINaPDannerNodeOptions
      :members:

   Leaky integrator with persistent sodium current.

   **Default Parameters:**

   .. list-table::
      :header-rows: 1

      * - Parameter
        - Default Value
        - Description
      * - c_m
        - 10.0
        - Membrane capacitance (pF)
      * - g_nap
        - 4.5
        - NaP conductance (nS)
      * - e_na
        - 50.0
        - Sodium reversal potential (mV)
      * - v1_2_m
        - -40.0
        - Half-activation voltage (mV)
      * - k_m
        - -6.0
        - Activation slope
      * - v1_2_h
        - -45.0
        - Half-inactivation voltage (mV)
      * - k_h
        - 4.0
        - Inactivation slope
      * - v1_2_t
        - -35.0
        - Threshold half-activation (mV)
      * - k_t
        - 15.0
        - Threshold slope
      * - g_leak
        - 4.5
        - Leak conductance (nS)
      * - e_leak
        - -62.5
        - Leak reversal potential (mV)
      * - tau_0
        - 80.0
        - Base time constant (ms)
      * - tau_max
        - 160.0
        - Maximum time constant (ms)

   Edge Options
   -----------

   .. autoclass:: farms_network.options.EdgeOptions
      :members:

   Configuration for synaptic connections.

   **Attributes:**

   - ``source`` (str): Source node name
   - ``target`` (str): Target node name
   - ``weight`` (float): Connection strength
   - ``type`` (str): Synapse type
   - ``parameters`` (EdgeParameterOptions): Model-specific parameters
   - ``visual`` (EdgeVisualOptions): Visualization settings

   Edge Visual Options
   ^^^^^^^^^^^^^^^^^

   .. autoclass:: farms_network.options.EdgeVisualOptions
      :members:

   **Default Values:**

   .. list-table::
      :header-rows: 1

      * - Parameter
        - Default Value
        - Description
      * - color
        - [1.0, 0.0, 0.0]
        - RGB values
      * - alpha
        - 1.0
        - Transparency
      * - label
        - ""
        - Display label
      * - layer
        - "background"
        - Rendering layer
      * - arrowstyle
        - "->"
        - Arrow appearance
      * - connectionstyle
        - "arc3,rad=0.1"
        - Connection curve
      * - linewidth
        - 1.5
        - Line thickness
      * - edgecolor
        - [0.0, 0.0, 0.0]
        - Border color

   Integration Options
   -----------------

   .. autoclass:: farms_network.options.IntegrationOptions
      :members:

   Numerical integration settings.

   **Default Values:**

   .. list-table::
      :header-rows: 1

      * - Parameter
        - Default Value
        - Description
      * - timestep
        - 0.001
        - Integration step (s)
      * - n_iterations
        - 1000
        - Number of steps
      * - integrator
        - "rk4"
        - Integration method
      * - method
        - "adams"
        - Solver method
      * - atol
        - 1e-12
        - Absolute tolerance
      * - rtol
        - 1e-6
        - Relative tolerance
      * - max_step
        - 0.0
        - Maximum step size
      * - checks
        - True
        - Enable validation

   Network Log Options
   -----------------

   .. autoclass:: farms_network.options.NetworkLogOptions
      :members:

   Logging configuration.

   **Default Values:**

   .. list-table::
      :header-rows: 1

      * - Parameter
        - Default Value
        - Description
      * - n_iterations
        - Required
        - Number of iterations to log
      * - buffer_size
        - n_iterations
        - Log buffer size
      * - nodes_all
        - False
        - Log all nodes

   Options
   =======

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

   LINaPDannerParameterOptions Class
   ---------------------------------
   .. autoclass:: farms_network.core.options.LINaPDannerParameterOptions
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
