Core Concepts
=============

Network Components
------------------

Node
^^^^
Basic unit representing a computational node, a neuron is a form of a node. A node receives n inputs and has n outputs.

Properties:

* Dynamics (ode)/computation
* Input integration
* Output generation

Available dynamics models:
  * Base
  * Relay
  * Linear
  * Relu
  * Oscillator
  * Hopf oscillator
  * Morphed oscillator
  * Matsuoka oscillator
  * Fitzhugh nagumo oscillator
  * Morris lecar oscillator
  * Leaky integrator
  * Leaky integrator (danner)
  * Leaky integrator with persistence sodium (danner)
  * Leaky integrator (daun)
  * Hodgkin-Huxley (daun)
  * Izhikevich
  * Hodgkin-Huxley
  * Custom (user-defined)

Edge
^^^^
Connection between any two nodes. Characteristics:

* Source node
* Target node
* Weight (connection strength)
* Type (excitatory/inhibitory/cholinergic)
* Parameters

Network
^^^^^^^
The primary container for the circuit simulation. Networks manage:

* Component organization and connectivity
* Simulation configuration
* State tracking and data collection
* Input/output handling

Simulation Elements
-------------------

Time Management
^^^^^^^^^^^^^^^
* ``dt``: Integration time step
* ``simulation_duration``: Total simulation time
* ``sampling_rate``: Data collection frequency

State Variables
^^^^^^^^^^^^^^^
* Membrane potentials
* Synaptic currents
* Ionic concentrations
* Firing rates
* Custom variables

Input Handling
^^^^^^^^^^^^^^
* External current injection
* Spike trains
* Continuous signals
* Stochastic inputs

Output and Analysis
^^^^^^^^^^^^^^^^^^^
* Membrane potential traces
* Spike times
* Population activity
* Network statistics
* Custom metrics

Configuration
-------------

Network Setup
^^^^^^^^^^^^^
.. code-block:: python

    net.configure(
        dt=0.1,                    # Time step (ms)
        simulation_duration=1000.0, # Duration (ms)
        sampling_rate=1.0,         # Recording frequency (ms)
        backend='cpu'              #  backend
    )

Node Configuration
^^^^^^^^^^^^^^^^^^
.. code node:: python

    neuron.configure(
        threshold=-55.0,           # Firing threshold (mV)
        reset_potential=-70.0,     # Reset potential (mV)
        refractory_period=2.0,     # Refractory period (ms)
        capacitance=1.0,           # Membrane capacitance (pF)
        leak_conductance=0.1       # Leak conductance (nS)
    )

Synapse Configuration
^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

Edge

        delay=1.0,                 # Transmission delay (ms)
        plasticity='stdp',         # Plasticity rule
        learning_rate=0.01         # Learning rate
    )
