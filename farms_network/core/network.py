""" Network """

from typing import List, Optional

import numpy as np

from ..models.factory import EdgeFactory, NodeFactory
from .data import NetworkData, NetworkLog
from .edge import Edge
from .network_cy import NetworkCy
from .node import Node
from .options import (EdgeOptions, IntegrationOptions, NetworkOptions,
                      NodeOptions)


class Network:
    """ Network class using composition with NetworkCy """

    def __init__(self, network_options: NetworkOptions):
        """ Initialize network with composition approach """
        self.options = network_options

        # Core network data and Cython implementation
        self.data = NetworkData.from_options(network_options)
        self.log = NetworkLog.from_options(network_options)

        self._network_cy = NetworkCy(
            nnodes=len(network_options.nodes),
            nedges=len(network_options.edges),
            data=self.data,
            log=self.log
        )

        # Python-level collections
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []

        # Setup the network
        self._setup_network()
        # self._setup_integrator()

        # Logs
        self.buffer_size: int = network_options.logs.buffer_size

        # Iteration
        if network_options.integration:
            self.timestep: float = network_options.integration.timestep
            self.iteration: int = 0
            self.n_iterations: int = network_options.integration.n_iterations

    def _setup_network(self):
        """ Setup network nodes and edges """
        # Create Python nodes
        nstates = 0
        for index, node_options in enumerate(self.options.nodes):
            python_node = self._generate_node(node_options)
            python_node._node_cy.ninputs = len(
                self.data.connectivity.node_indices[
                    self.data.connectivity.index_offsets[index]:self.data.connectivity.index_offsets[index+1]
                ]
            ) if self.data.connectivity.index_offsets else 0
            nstates += python_node.nstates
            self.nodes.append(python_node)

        # Create Python edges
        for edge_options in self.options.edges:
            python_edge = self._generate_edge(edge_options)
            self.edges.append(python_edge)

        self._network_cy.nstates = nstates

        # Pass Python nodes/edges to Cython layer for C struct setup
        self._network_cy.setup_network(self.options, self.data, self.nodes, self.edges)

        # Initialize states
        self._initialize_states()

    def _setup_integrator(self):
        """ Setup numerical integrators """
        self._network_cy.setup_integrator(self.options)

    def _initialize_states(self):
        """ Initialize node states from options """
        for j, node_opts in enumerate(self.options.nodes):
            if node_opts.state:
                for state_index, index in enumerate(
                    range(self.data.states.indices[j], self.data.states.indices[j+1])
                ):
                    self.data.states.array[index] = node_opts.state.initial[state_index]

    @staticmethod
    def _generate_node(node_options: NodeOptions) -> Node:
        """ Generate a node from options """
        NodeClass = NodeFactory.create(node_options.model)
        return NodeClass.from_options(node_options)

    @staticmethod
    def _generate_edge(edge_options: EdgeOptions) -> Edge:
        """ Generate an edge from options """
        EdgeClass = EdgeFactory.create(edge_options.model)
        return EdgeClass.from_options(edge_options)

    def get_ode_func(self):
        """ Get ODE function for external integration """
        return self._network_cy.ode_func

    # Delegate properties to Cython implementation
    @property
    def nnodes(self) -> int:
        return self._network_cy.nnodes

    @property
    def nedges(self) -> int:
        return self._network_cy.nedges

    @property
    def nstates(self) -> int:
        return self._network_cy.nstates

    def step(self):
        """ Step the network simulation """
        self._network_cy.step()
        self.iteration += 1

    def run(self, n_iterations: Optional[int] = None):
        """ Run the network for n_iterations """
        if n_iterations is None:
            n_iterations = self.n_iterations

        for _ in range(n_iterations):
            self.step()

    # Factory methods
    @classmethod
    def from_options(cls, options: NetworkOptions):
        """ Initialize network from NetworkOptions """
        return cls(options)

    def to_options(self) -> NetworkOptions:
        """ Return NetworkOptions from network """
        return self.options
