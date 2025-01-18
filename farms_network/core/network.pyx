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
from ..noise.ornstein_uhlenbeck import OrnsteinUhlenbeck
from .data import NetworkData, NetworkStates

from .data_cy cimport (NetworkConnectivityCy, NetworkDataCy, NetworkNoiseCy,
                       NetworkStatesCy)

from .options import (EdgeOptions, IntegrationOptions, NetworkOptions,
                      NodeOptions)


cdef inline void ode(
    double time,
    double[:] states_arr,
    NetworkDataCy data,
    NetworkCy* c_network,
    double[:] node_outputs_tmp,
) noexcept:
    """ C Implementation to compute full network state """
    cdef unsigned int j, nnodes

    cdef NodeCy __node
    cdef NodeCy** c_nodes = c_network.c_nodes
    cdef EdgeCy** c_edges = c_network.c_edges
    nnodes = c_network.nnodes

    # It is important to use the states passed to the function and not from the data.states
    cdef double* states = &states_arr[0]
    cdef unsigned int* states_indices = &data.states.indices[0]

    cdef double* derivatives = &data.derivatives.array[0]
    cdef unsigned int* derivatives_indices = &data.derivatives.indices[0]

    cdef double* external_input = &data.external_inputs.array[0]
    cdef double* outputs = &data.outputs.array[0]

    cdef double* noise = &data.noise.outputs[0]

    cdef unsigned int* input_neurons = &data.connectivity.sources[0]
    cdef double* weights = &data.connectivity.weights[0]
    cdef unsigned int* input_neurons_indices = &data.connectivity.indices[0]

    cdef double* node_outputs_tmp_ptr = &node_outputs_tmp[0]

    for j in range(nnodes):
        __node = c_nodes[j][0]
        if __node.is_statefull:
            __node.ode(
                time,
                states + states_indices[j],
                derivatives + derivatives_indices[j],
                external_input[j],
                outputs,
                input_neurons + input_neurons_indices[j],
                weights + input_neurons_indices[j],
                noise[j],
                c_nodes[j],
                c_edges + input_neurons_indices[j],
            )
        node_outputs_tmp_ptr[j] = __node.output(
            time,
            states + states_indices[j],
            external_input[j],
            outputs,
            input_neurons + input_neurons_indices[j],
            weights + input_neurons_indices[j],
            c_nodes[j],
            c_edges + input_neurons_indices[j],
        )


cdef inline void logger(
    int iteration,
    NetworkDataCy data,
    NetworkCy* c_network
) noexcept:
    cdef unsigned int nnodes = c_network.nnodes
    cdef unsigned int j
    cdef double* states_ptr = &data.states.array[0]
    cdef unsigned int[:] state_indices = data.states.indices
    cdef double[:] outputs = data.outputs.array
    cdef double* outputs_ptr = &data.outputs.array[0]
    cdef double[:] external_inputs = data.external_inputs.array
    cdef NodeDataCy node_data
    cdef double[:] node_states
    cdef int state_idx, start_idx, end_idx, state_iteration
    cdef NodeDataCy[:] nodes_data = data.nodes
    for j in range(nnodes):
        # Log states
        start_idx = state_indices[j]
        end_idx = state_indices[j+1]
        state_iteration = 0
        node_states = nodes_data[j].states.array[iteration]
        for state_idx in range(start_idx, end_idx):
            node_states[state_iteration] = states_ptr[state_idx]
            state_iteration += 1
        nodes_data[j].output.array[iteration] = outputs_ptr[j]
        nodes_data[j].external_input.array[iteration] = external_inputs[j]


cdef inline void _noise_states_to_output(
    double[:] states,
    unsigned int[:] indices,
    double[:] outputs,
) noexcept:
    """ Copy noise states data to noise outputs """
    cdef int n_indices = indices.shape[0]
    cdef int index
    for index in range(n_indices):
        outputs[indices[index]] = states[index]


cdef class Network(ODESystem):
    """ Python interface to Network ODE """

    def __cinit__(self, network_options: NetworkOptions):
        """ C initialization for manual memory allocation """
        self.c_network = <NetworkCy*>malloc(sizeof(NetworkCy))
        if self.c_network is NULL:
            raise MemoryError("Failed to allocate memory for Network")
        self.c_network.nnodes = len(network_options.nodes)
        self.c_network.nedges = len(network_options.edges)
        # Allocate memory for c-node structs
        self.c_network.c_nodes = <NodeCy**>malloc(self.nnodes * sizeof(NodeCy *))
        if self.c_network.c_nodes is NULL:
            raise MemoryError("Failed to allocate memory for Network nodes")
        # Allocation memory for c-edge structs
        self.c_network.c_edges = <EdgeCy**>malloc(self.nedges * sizeof(EdgeCy *))
        if self.c_network.c_edges is NULL:
            raise MemoryError("Failed to allocate memory for Network edges")

    def __init__(self, network_options: NetworkOptions):
        """ Initialize """

        super().__init__()
        self.data = <NetworkDataCy>NetworkData.from_options(network_options)

        self.nodes = []
        self.edges = []
        self.nodes_output_data = []
        self.__tmp_node_outputs = np.zeros((self.c_network.nnodes,))
        self.setup_network(network_options, self.data)

        # Integration options
        self.n_iterations: int = network_options.integration.n_iterations
        self.timestep: int = network_options.integration.timestep
        self.iteration: int = 0
        self.buffer_size: int = network_options.logs.buffer_size

        # Set the seed for random number generation
        random_seed = network_options.random_seed
        # np.random.seed(random_seed)

    def __dealloc__(self):
        """ Deallocate any manual memory as part of clean up """
        if self.c_network.c_nodes is not NULL:
            free(self.c_network.c_nodes)
        if self.c_network.c_edges is not NULL:
            free(self.c_network.c_edges)
        if self.c_network is not NULL:
            free(self.c_network)

    @classmethod
    def from_options(cls, options: NetworkOptions):
        """ Initialize network from NetworkOptions """
        return cls(options)

    def to_options(self):
        """ Return NetworkOptions from network """
        return self.options

    def setup_integrator(self, network_options: NetworkOptions):
        """ Setup integrator for neural network """
        # Setup ODE numerical integrator
        integration_options = network_options.integration
        cdef double timestep = integration_options.timestep
        self.ode_integrator = RK4Solver(self.c_network.nstates, timestep)
        # Setup SDE numerical integrator for noise models if any
        noise_options = []
        for node in network_options.nodes:
            if node.noise is not None:
                if node.noise.is_stochastic:
                    noise_options.append(node.noise)

        self.sde_system = OrnsteinUhlenbeck(noise_options)
        self.sde_integrator = EulerMaruyamaSolver(len(noise_options), timestep)

    def setup_network(self, options: NetworkOptions, data: NetworkData):
        """ Setup network """

        connectivity = data.connectivity
        cdef unsigned int __nstates = 0
        cdef unsigned int index
        cdef NodeCy* c_node
        cdef Node node
        for index, node_options in enumerate(options.nodes):
            self.nodes.append(self.generate_node(node_options))
            node = <Node> self.nodes[index]
            c_node = (<NodeCy*>node.c_node)
            c_node.ninputs = len(
                connectivity.sources[
                    connectivity.indices[index]:connectivity.indices[index+1]
                ]
            ) if connectivity.indices else 0
            self.c_network.c_nodes[index] = c_node
            __nstates += node_options._nstates
        self.c_network.nstates = __nstates

        cdef EdgeCy* c_edge
        cdef Edge edge
        for index, edge_options in enumerate(options.edges):
            self.edges.append(self.generate_edge(edge_options, options.nodes))
            edge = <Edge> self.edges[index]
            c_edge = (<EdgeCy*>edge.c_edge)
            self.c_network.c_edges[index] = c_edge

        # Initial states data
        # Initialize states
        for j, node_opts in enumerate(options.nodes):
            if node_opts.state:
                for state_index, index in enumerate(
                        range(data.states.indices[j], data.states.indices[j+1])
                ):
                    data.states.array[index] = node_opts.state.initial[state_index]

    @staticmethod
    def generate_node(node_options: NodeOptions):
        """ Generate a node from options """
        Node = NodeFactory.create(node_options.model)
        node = Node.from_options(node_options)
        return node

    @staticmethod
    def generate_edge(edge_options: EdgeOptions, nodes_options):
        """ Generate a edge from options """
        target = nodes_options[nodes_options.index(edge_options.target)]
        Edge = EdgeFactory.create(target.model)
        edge = Edge.from_options(edge_options)
        return edge

    @property
    def nnodes(self):
        """ Number of nodes in the network """
        return self.c_network.nnodes

    @property
    def nedges(self):
        """ Number of edges in the network """
        return self.c_network.nedges

    @property
    def nstates(self):
        """ Number of states in the network """
        return self.c_network.nstates

    cpdef void step(self):
        """ Step the network state """
        # cdef NetworkDataCy data = self.data
        cdef SDESystem sde_system = self.sde_system
        cdef EulerMaruyamaSolver sde_integrator = self.sde_integrator

        sde_integrator.step(
            sde_system,
            (self.iteration%self.buffer_size)*self.timestep,
            self.data.noise.states
        )
        _noise_states_to_output(
            self.data.noise.states,
            self.data.noise.indices,
            self.data.noise.outputs
        )
        self.ode_integrator.step(
            self,
            (self.iteration%self.buffer_size)*self.timestep,
            self.data.states.array
        )
        # Logging
        # TODO: Use network options to check global logging flag
        logger((self.iteration%self.buffer_size), self.data, self.c_network)
        self.iteration += 1

    cdef void evaluate(self, double time, double[:] states, double[:] derivatives) noexcept:
        """ Evaluate the ODE """
        # Update noise model
        cdef NetworkDataCy data = <NetworkDataCy> self.data

        ode(time, states, data, self.c_network, self.__tmp_node_outputs)
        data.outputs.array[:] = self.__tmp_node_outputs
        derivatives[:] = data.derivatives.array
