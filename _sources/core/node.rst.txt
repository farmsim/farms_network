Node
====

This documentation describes the Node structure and Python interface provided in the `node.pyx` and `node.pxd` files.

Contents
--------

- Node C Structure
- PyNode Python Class
- Functions (ODE and Output)

Node C Structure
----------------

The Node C structure defines the internal state and behavior of a node in a dynamical system.
It contains generic attributes like state variables, inputs, and parameters.
All nodes are many-inputs-single-output (MISO).
The simplest case would be a node with one input and one input.
A node can have N-states that will be integrated by a numerical integrator over time.
A stateless node will have zero states and is useful in using the node as a transfer function.

.. automodule:: farms_network.core.node
   :platform: Unix, Windows
   :synopsis: Provides Node C-Structure and Python interface for nodes in a dynamical system.
   :members:
   :show-inheritance:
   :private-members:
   :special-members:
   :noindex:
