Node
====

This documentation describes the Node structure and Python interface provided in the `node_cy.pyx` and `node_cy.pxd` files.

Contents
--------

- Node C Structure
- Node Python Class
- Functions (ODE and Output)

Node C Structure
----------------

The Node C structure defines the internal state and behavior of a node in a dynamical system.
It contains generic attributes like state variables, inputs, and parameters.
All nodes are many-inputs-single-output (MISO).
The simplest case would be a node with one input and one input.
A node can have N-states that will be integrated by a numerical integrator over time.
A stateless node will have zero states and is useful in using the node as a transfer function.


Node
====

.. autoclass:: farms_network.node.Node
   :members:

The Node class provides a high-level interface for neural network nodes.

Constructor
----------

.. code-block:: python

    Node(name: str, **kwargs)

**Parameters:**

- ``name`` (str): Unique identifier for the node
- ``**kwargs``: Additional configuration parameters passed to NodeCy

Class Methods
------------

from_options
^^^^^^^^^^^

Create a node from configuration options.

.. code-block:: python

    @classmethod
    def from_options(cls, node_options: NodeOptions) -> Node

**Parameters:**

- ``node_options`` (NodeOptions): Configuration options

**Returns:**

- Node: Configured node instance

Examples
--------

Creating a node:

.. code-block:: python

    # Direct instantiation
    node = Node("neuron1")

    # From options
    options = NodeOptions(name="neuron1")
    node = Node.from_options(options)


.. automodule:: farms_network.core.node_cy
   :platform: Unix, Windows
   :synopsis: Provides Node C-Structure for nodes in a dynamical system.
   :members:
   :show-inheritance:
   :noindex:
