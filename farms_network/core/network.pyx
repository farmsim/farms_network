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

cimport numpy as cnp
import numpy as np

from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strdup

from ..models.factory import NodeFactory

from ..models.li_danner cimport LIDannerNodeParameters

from .data import NetworkData, NetworkStates

from .data_cy cimport NetworkDataCy, NetworkStatesCy
from .node cimport Node, PyNode

from tqdm import tqdm

from .options import (EdgeOptions, IntegrationOptions, NetworkOptions,
                      NodeOptions)


cpdef rk4(double time, cnp.ndarray[double, ndim=1] state, func, double step_size):
    """ Runge-kutta order 4 integrator """

    K1 = np.array(func(time, state))
    K2 = np.array(func(time + step_size/2, state + (step_size/2 * K1)))
    K3 = np.array(func(time + step_size/2, state + (step_size/2 * K2)))
    K4 = np.array(func(time + step_size, state + (step_size * K3)))
    state = state + (K1 + 2*K2 + 2*K3 + K4)*(step_size/6)
    time += step_size
    return state


cdef void ode(
    double time,
    NetworkDataCy data,
    Network* network,
    double[:] tmp_node_outputs
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

    cdef double* tmp_node_outputs_ptr = &tmp_node_outputs[0]

    for j in range(nnodes):
        __node = nodes[j][0]
        if __node.is_statefull:
            __node.ode(
                time,
                states + states_indices[j],
                derivatives + derivatives_indices[j],
                external_input,
                outputs,
                input_neurons + input_neurons_indices[j],
                weights + input_neurons_indices[j],
                nodes[j]
            )
        tmp_node_outputs_ptr[j] = __node.output(
            time,
            states + states_indices[j],
            external_input,
            outputs,
            input_neurons + input_neurons_indices[j],
            weights + input_neurons_indices[j],
            nodes[j]
        )
    # Update all node outputs data for next iteration
    data.outputs.array[:] = tmp_node_outputs[:]


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
        self.__tmp_node_outputs = np.zeros((self.network.nnodes,))
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

    def setup_integrator(options: IntegrationOptions):
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

    @staticmethod
    def generate_node(node_options: NodeOptions):
        """ Generate a node from options """
        Node = NodeFactory.generate_node(node_options.model)
        node = Node.from_options(node_options)
        return node

    cpdef double[:] ode(self, double time, double[::1] states):
        """ ODE Wrapper for numerical integrators  """
        self.data.states.array[:] = states[:]
        ode(time, self.data, self.network, self.__tmp_node_outputs)
        return self.data.derivatives.array

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
