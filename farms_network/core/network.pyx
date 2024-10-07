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

from typing import Iterable

from ..models.factory import NodeFactory
from ..models.li_danner cimport LIDannerNodeParameters
from .data import NetworkData, NetworkStates
from .data_cy cimport NetworkDataCy, NetworkStatesCy

from .node cimport Node, PyNode

from tqdm import tqdm

from .options import EdgeOptions, NetworkOptions, NodeOptions


cdef enum state_conv:
    state0 = 0
    state1 = 1


cdef void li_ode(
    double time,
    double* states,
    double* derivatives,
    double external_input,
    double* network_outputs,
    unsigned int* input_neurons,
    double* input_weights,
    Node* node
) noexcept:
    """ ODE """
    # Parameters
    cdef LIDannerNodeParameters params = (<LIDannerNodeParameters*> node[0].parameters)[0]

    # States
    cdef double state_v = states[<int>state_conv.state0]
    # printf("states: %f \n", state_v)

    # Ileak
    cdef double i_leak = params.g_leak * (state_v - params.e_leak)

    # Node inputs
    cdef double _sum = 0.0
    cdef unsigned int j
    cdef double _node_out
    cdef double res

    cdef double _input
    cdef double _weight

    cdef unsigned int ninputs = node[0].ninputs
    for j in range(ninputs):
        _input = network_outputs[input_neurons[j]]
        _weight = input_weights[j]
        # _sum += node_inputs_eval_c(_input, _weight)

    # dV
    cdef double i_noise = 0.0
    states[<int>state_conv.state0] = (-(i_leak + i_noise + _sum)/params.c_m)


cdef void ode(
    double time,
    unsigned int iteration,
    NetworkDataCy data,
    Network* network,
) noexcept:
    """ C Implementation to compute full network state """
    cdef int j, step, steps, nnodes
    steps = 4

    cdef Node __node
    cdef Node** nodes = network.nodes
    nnodes = network.nnodes

    cdef double* states = &data.states.array[0]
    cdef unsigned int* states_indices = &data.states.indices[0]

    cdef double* derivatives = &data.derivatives.array[0]
    cdef unsigned int* derivatives_indices = &data.derivatives.indices[0]

    cdef double external_input = 0.0
    cdef double* outputs = &data.outputs.array[0]

    cdef unsigned int* input_neurons = &data.connectivity.sources[0]
    cdef double* weights = &data.connectivity.weights[0]
    cdef unsigned int* input_neurons_indices = &data.connectivity.indices[0]

    # cdef double[::1] node_states
    # cdef double[::1] node_derivatives

    # node_derivatives = states[derivatives_indices[0]:derivatives_indices[1]]

    for step in range(steps):
        for j in range(nnodes):
            # node_states = states[states_indices[0]:states_indices[1]]
            # (nodes[j][0]).ode(
            #     time, node_states, node_derivatives, time, node_states, node_states, node_states, nodes[j]
            # )
            li_ode(
                time,
                states + states_indices_ptr[j],
                derivatives + derivatives_indices[j],
                external_input,
                outputs,
                input_neurons + input_neurons_indices[j],
                weights + input_neurons_indices[j],
                nodes[j]
            )
            # if __node.statefull:
            #     __node.output(0.0, states, 0.0, arr, arr, arr, __node)
            # else:
            #     __node.output(0.0, arr, 0.0, arr, arr, arr, __node)


cdef class PyNetwork:
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

    def __init__(self, network_options):
        """ Initialize """

        super().__init__()
        self.data = <NetworkDataCy>NetworkData.from_options(network_options)
        self.pynodes = []
        self.setup_network(network_options, self.data)

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

    def setup_integrator(options: IntegratorOptions):
        """ Setup integrator for neural network """
        ...

    def setup_network(self, options: NetworkOptions, data: NetworkData):
        """ Setup network """

        cdef Node* c_node

        connectivity = data.connectivity
        for index, node_options in enumerate(options.nodes):
            self.pynodes.append(self.generate_node(node_options))
            pyn = <PyNode> self.pynodes[index]
            c_node = (<Node*>pyn.node)
            c_node.ninputs = len(
                connectivity.sources[connectivity.indices[index]:connectivity.indices[index+1]]
            )
            self.network.nodes[index] = c_node

    def generate_node(self, node_options: NodeOptions):
        """ Generate a node from options """
        Node = NodeFactory.generate_node(node_options.model)
        node = Node.from_options(node_options)
        return node

    cpdef void ode(self, double time, double[:] states):
        """ ODE Wrapper for numerical integrators  """
        self.data.states.array[:] = states[:]
        cdef int iteration = 0
        ode(time, iteration, self.data, self.network)

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

    # def step(self, time, states):
    #     """ Update the network """
    #     self.network.ode(time, 0, states, states, self.c_nodes)
    #     dstates = None
    #     return dstates

    # cpdef void step(self)
    #     """ Network step function """
    #     ...
