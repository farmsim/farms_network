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

from .data import NetworkData, NetworkStates

from .node cimport PyNode

from tqdm import tqdm

from .options import EdgeOptions, NetworkOptions, NodeOptions


cdef void ode(
    double time,
    unsigned int iteration,
    double[:] states,
    double[:] derivatives,
    Network network,
    NetworkDataCy data,
) noexcept:
    """ C Implementation to compute full network state """
    cdef Node __node
    cdef Node **nodes = network.nodes
    cdef unsigned int t, j
    cdef double[:] arr = None
    printf("\n")
    cdef double[:] _states = data.states.array
    cdef long unsigned int nnodes = network.nnodes
    for t in range(nnodes):
        printf("%i \t", t)
        for j in range(100):
            __node = nodes[j][0]
            __node.ode(0.0, _states, arr, 0.0, arr, arr, arr, __node)
            if __node.statefull:
                __node.output(0.0, states, 0.0, arr, arr, arr, __node)
            else:
                __node.output(0.0, arr, 0.0, arr, arr, arr, __node)


cdef class PyNetwork:
    """ Python interface to Network ODE """

    def __cinit__(self):
        """ C initialization for manual memory allocation """
        self.network = <Network*>malloc(sizeof(Network))
        if self.network is NULL:
            raise MemoryError("Failed to allocate memory for Network")
        self.network.ode = ode

    def __init__(self, network_options):
        """ Initialize """
        nodes = network_options.nodes
        edges = network_options.edges
        self.network.nnodes = len(nodes)
        self.network.nedges = len(edges)

        self.data = NetworkData.from_options(network_options)
        # Allocate memory for c-node structs
        self.network.nodes = <Node **>malloc(self.nnodes * sizeof(Node *))
        cdef Node *c_node
        self.nodes = []

        for n in range(self.nnodes):
            self.nodes.append(PyNode(f"{n}"))
            pyn = <PyNode> self.nodes[n]
            c_node = (<Node*>pyn.node)
            self.network.nodes[n] = c_node

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

    cpdef void ode(self, double time, double[:] states):
        cdef double[:] _states = self.data.states.array
        cdef UITYPEv1 _indices = self.data.states.indices
        self.network.ode(
            time, 0, _states, states, self.network[0], self.data
        )


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
