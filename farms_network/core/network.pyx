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

import numpy as np

from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strdup

# from ..models.li_danner cimport PyLIDannerNode

from .options import NetworkOptions

from .node cimport PyNode

from tqdm import tqdm


cdef void ode(
    double time,
    unsigned int iteration,
    double[:] states,
    double[:] derivatives,
    Node **nodes,
) noexcept:
    """ C Implementation to compute full network state """
    cdef Node __node
    cdef unsigned int t, j
    for t in range(int(1000*1e3)):
        for j in range(100):
            __node = nodes[j][0]


cdef class PyNetwork:
    """ Python interface to Network ODE """

    def __cinit__(self, nnodes: int):
        """ C initialization for manual memory allocation """
        self.nnodes = nnodes
        self._network = <Network*>malloc(sizeof(Network))
        if self._network is NULL:
            raise MemoryError("Failed to allocate memory for Network")
        self._network.ode = ode
        self.c_nodes = <Node **>malloc(nnodes * sizeof(Node *))
        # if self._network.nodes is NULL:
        #     raise MemoryError("Failed to allocate memory for nodes in Network")

        self.nodes = []
        cdef Node *c_node

        for n in range(self.nnodes):
            # self.nodes.append(PyLIDannerNode(f"{n}", 0))
            # pyn = <PyLIDannerNode> self.nodes[n]
            # c_node = (<Node*>pyn._node)
            # self.c_nodes[n] = c_node

            self.nodes.append(PyNode(f"{n}", 0))
            pyn = <PyNode> self.nodes[n]
            c_node = (<Node*>pyn._node)
            self.c_nodes[n] = c_node

    def __dealloc__(self):
        """ Deallocate any manual memory as part of clean up """
        if self._network.nodes is not NULL:
            free(self._network.nodes)
        if self._network is not NULL:
            free(self._network)

    @classmethod
    def from_options(cls, options: NetworkOptions):
        """ Initialize network from NetworkOptions """
        options
        return cls

    def setup_integrator(options: IntegratorOptions):
        """ Setup integrator for neural network """
        ...

    cpdef void ode(self, double time, double[:] states):
        self._network.ode(time, 0, states, states, self.c_nodes)

    cpdef void step(self):
        """ Network step function """
        ...
