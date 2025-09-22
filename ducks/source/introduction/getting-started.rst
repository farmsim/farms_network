Getting Started
===============

This guide demonstrates basic usage of farms_network through a simple example.

Basic Example
------------

Let's create a simple neural network that simulates a basic reflex circuit:

.. code-block:: python

    from farms_network import Network, Node, Edge
    import numpy as np

    # Create network
    net = Network('reflex_circuit')

    # Add neurons
    sensory = Node('sensory', dynamics='leaky_integrator')
    inter = Node('interneuron', dynamics='leaky_integrator')
    motor = Node('motor', dynamics='leaky_integrator')

    # Add neurons to network
    net.add_neurons([sensory, inter, motor])

    # Create synaptic connections
    syn1 = Synapse('sensory_to_inter', sensory, inter, weight=0.5)
    syn2 = Synapse('inter_to_motor', inter, motor, weight=0.8)

    # Add synapses to network
    net.add_synapses([syn1, syn2])

    # Configure simulation parameters
    net.configure(dt=0.1, simulation_duration=10.0)

    # Add input stimulus
    stimulus = np.sin(np.linspace(0, 10, 100))
    net.set_external_input(sensory, stimulus)

    # Run simulation
    results = net.simulate()

    # Plot results
    net.plot_results(results)

Key Concepts
-----------

* **Network**: The main container for your neural circuit
* **Node**: Represents a neural unit with specific dynamics
* **Synapse**: Defines connections between neur
