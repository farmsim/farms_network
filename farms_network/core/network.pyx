"""
-----------------------------------------------------------------------
Copyright 2018-2020 Jonathan Arreguit, Shravan Tata Ramalingasetty
Copyright 2018 BioRobotics Laboratory, École polytechnique fédérale de Lausanne

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-----------------------------------------------------------------------
"""

include "types.pxd"

import numpy as np

from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strdup

from ..models.factory import EdgeFactory, NodeFactory

from ..models.li_danner cimport LIDannerNodeParameters

from .data import NetworkData, NetworkStates

from .data_cy cimport NetworkDataCy, NetworkStatesCy, NetworkConnectivityCy
from .edge cimport Edge, PyEdge
from .node cimport Node, PyNode

from .options import (EdgeOptions, IntegrationOptions, NetworkOptions,
                      NodeOptions)


cdef inline void ode(
    double time,
    NetworkDataCy data,
    Network* network,
    double[:] node_outputs_tmp,
) noexcept:
    """ C Implementation to compute full network state """
    cdef unsigned int j, nnodes

    cdef Node __node
    cdef Node** nodes = network.nodes
    cdef Edge** edges = network.edges
    nnodes = network.nnodes

    cdef double* states = &data.states.array[0]
    cdef unsigned int* states_indices = &data.states.indices[0]

    cdef double* derivatives = &data.derivatives.array[0]
    cdef unsigned int* derivatives_indices = &data.derivatives.indices[0]

    cdef double* external_input = &data.external_inputs.array[0]
    cdef double* outputs = &data.outputs.array[0]

    cdef unsigned int* input_neurons = &data.connectivity.sources[0]
    cdef double* weights = &data.connectivity.weights[0]
    cdef unsigned int* input_neurons_indices = &data.connectivity.indices[0]

    cdef double* node_outputs_tmp_ptr = &node_outputs_tmp[0]

    for j in range(nnodes):
        __node = nodes[j][0]
        if __node.is_statefull:
            __node.ode(
                time,
                states + states_indices[j],
                derivatives + derivatives_indices[j],
                external_input[j],
                outputs,
                input_neurons + input_neurons_indices[j],
                weights + input_neurons_indices[j],
                nodes[j],
                edges + input_neurons_indices[j],
            )
        node_outputs_tmp_ptr[j] = __node.output(
            time,
            states + states_indices[j],
            external_input[j],
            outputs,
            input_neurons + input_neurons_indices[j],
            weights + input_neurons_indices[j],
            nodes[j],
            edges + input_neurons_indices[j],
        )


cdef class PyNetwork(ODESystem):
    """ Python interface to Network ODE """

    def __cinit__(self, network_options: NetworkOptions):
        """ C initialization for manual memory allocation """
        self.network = <Network*>malloc(sizeof(Network))
        if self.network is NULL:
            raise MemoryError("Failed to allocate memory for Network")
        self.network.nnodes = len(network_options.nodes)
        self.network.nedges = len(network_options.edges)
        # Allocate memory for c-node structs
        self.network.nodes = <Node**>malloc(self.nnodes * sizeof(Node *))
        if self.network.nodes is NULL:
            raise MemoryError("Failed to allocate memory for Network nodes")
        # Allocation memory for c-edge structs
        self.network.edges = <Edge**>malloc(self.nedges * sizeof(Edge *))
        if self.network.edges is NULL:
            raise MemoryError("Failed to allocate memory for Network edges")

    def __init__(self, network_options: NetworkOptions):
        """ Initialize """

        super().__init__()
        self.data = <NetworkDataCy>NetworkData.from_options(network_options)
        self.pynodes = []
        self.pyedges = []
        self.nodes_output_data = []
        self.__tmp_node_outputs = np.zeros((self.network.nnodes,))
        self.setup_network(network_options, self.data)

        # Integration options
        self.n_iterations: int = network_options.integration.n_iterations
        self.timestep: int = network_options.integration.timestep
        self.iteration: int = 0
        self.buffer_size: int = network_options.logs.buffer_size

    def __dealloc__(self):
        """ Deallocate any manual memory as part of clean up """
        if self.network.nodes is not NULL:
            free(self.network.nodes)
        if self.network is not NULL:
            free(self.network)

    @classmethod
    def from_options(cls, options: NetworkOptions):
        """ Initialize network from NetworkOptions """
        return cls(options)

    def to_options(self):
        """ Return NetworkOptions from network """
        return self.options

    def setup_integrator(self, options: IntegrationOptions):
        """ Setup integrator for neural network """
        cdef double timestep = options.timestep
        self.integrator = RK4Solver(self.network.nstates, timestep)

    def setup_network(self, options: NetworkOptions, data: NetworkData):
        """ Setup network """

        cdef Node* c_node

        connectivity = data.connectivity
        cdef unsigned int __nstates = 0
        cdef unsigned int index
        cdef PyNode pyn
        cdef PyEdge pye
        for index, node_options in enumerate(options.nodes):
            self.pynodes.append(self.generate_node(node_options))
            pyn = <PyNode> self.pynodes[index]
            c_node = (<Node*>pyn.node)
            c_node.ninputs = len(
                connectivity.sources[connectivity.indices[index]:connectivity.indices[index+1]]
            ) if connectivity.indices else 0
            self.network.nodes[index] = c_node
            __nstates += node_options._nstates
        self.network.nstates = __nstates

        cdef Edge* c_edge
        for index, edge_options in enumerate(options.edges):
            self.pyedges.append(self.generate_edge(edge_options, options.nodes))
            pye = <PyEdge> self.pyedges[index]
            c_edge = (<Edge*>pye.edge)
            self.network.edges[index] = c_edge

    @staticmethod
    def generate_node(node_options: NodeOptions):
        """ Generate a node from options """
        Node = NodeFactory.generate_node(node_options.model)
        node = Node.from_options(node_options)
        return node

    @staticmethod
    def generate_edge(edge_options: EdgeOptions, nodes_options):
        """ Generate a edge from options """
        target = nodes_options[nodes_options.index(edge_options.target)]
        Edge = EdgeFactory.generate_edge(target.model)
        edge = Edge.from_options(edge_options)
        return edge

    @property
    def nnodes(self):
        """ Number of nodes in the network """
        return self.network.nnodes

    @property
    def nedges(self):
        """ Number of edges in the network """
        return self.network.nedges

    @property
    def nstates(self):
        """ Number of states in the network """
        return self.network.nstates

    cpdef void step(self) noexcept:
        """ Step the network state """
        self.iteration += 1
        self.integrator.step(
            self, self.iteration*self.timestep, self.data.states.array
        )
        PyNetwork.logging(self.iteration, self.data, self.network)

    @staticmethod
    cdef void logging(int iteration, NetworkDataCy data, Network* network) noexcept:
        """ Log network data """
        cdef int j, nnodes
        nnodes = network.nnodes

        # cdef double[:] states = data.states.array
        cdef double* states_ptr = &data.states.array[0]
        cdef unsigned int[:] state_indices = data.states.indices
        cdef int state_idx, start_idx, end_idx, state_iteration

        # cdef double[:] derivatives = data.derivatives.array
        cdef double* derivatives_ptr = &data.derivatives.array[0]

        cdef double[:] outputs = data.outputs.array

        cdef double[:] external_inputs = data.external_inputs.array
        cdef NodeDataCy node_data
        for j in range(nnodes):
            # Log states
            node_data = <NodeDataCy>data.nodes[j]
            start_idx = state_indices[j]
            end_idx = state_indices[j+1]
            state_iteration = 0
            for state_idx in range(start_idx, end_idx):
                node_data.states.array[iteration, state_iteration] = states_ptr[state_idx]
                node_data.derivatives.array[iteration, state_iteration] = derivatives_ptr[state_idx]
                state_iteration += 1
            node_data.output.array[iteration] = outputs[j]
            node_data.external_input.array[iteration] = external_inputs[j]

    cdef void evaluate(self, double time, double[:] states, double[:] derivatives) noexcept:
        """ Evaluate the ODE """
        # cdef NetworkDataCy data = <NetworkDataCy> self.data
        self.data.states.array[:] = states[:]
        ode(time, self.data, self.network, self.__tmp_node_outputs)
        self.data.outputs.array[:] = self.__tmp_node_outputs
        derivatives[:] = self.data.derivatives.array
