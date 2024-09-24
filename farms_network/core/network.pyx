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

from libc.stdlib cimport free, malloc
from libc.string cimport strdup

from ..models.li_danner cimport PyLIDannerNode
from .node cimport Node, PyNode

from tqdm import tqdm


cdef class PyNetwork:
    """ Python interface to Network ODE """

    def __cinit__(self, nnodes: int):
        """ C initialization for manual memory allocation """
        self.nnodes = nnodes
        self._network = <Network*>malloc(sizeof(Network))
        if self._network is NULL:
            raise MemoryError("Failed to allocate memory for Network")
        self.c_nodes = <Node **>malloc(nnodes * sizeof(Node *))
        # if self._network.nodes is NULL:
        #     raise MemoryError("Failed to allocate memory for nodes in Network")

        self.nodes = []
        cdef Node *c_node
        cdef PyLIDannerNode pyn

        for n in range(self.nnodes):
            self.nodes.append(PyLIDannerNode(f"{n}", 0))
            pyn = <PyLIDannerNode> self.nodes[n]
            c_node = (<Node*>pyn._node)
            self.c_nodes[n] = c_node

    def __dealloc__(self):
        """ Deallocate any manual memory as part of clean up """
        if self._network.nodes is not NULL:
            free(self._network.nodes)
        if self._network is not NULL:
            free(self._network)

    cpdef void test(self, data):
        cdef double[:] states = data.states.array[0, :]
        cdef double[:] dstates = np.empty((2,))
        cdef double[:] inputs = np.empty((10,))
        cdef double[:] weights = np.empty((10,))
        cdef double[:] noise = np.empty((10,))

        cdef Node **nodes = self.c_nodes
        cdef unsigned int t, j
        for t in tqdm(range(int(1000*1e3))):
            for j in range(self.nnodes):
                nodes[j][0].nstates
                nodes[j][0].ode_rhs_c(
                    0.0,
                    states,
                    dstates,
                    inputs,
                    weights,
                    noise,
                    nodes[j][0]
                )

    def step(self):
        """ Network step function """
        self._network.step()
