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


from .neuron cimport Neuron, PyNeuron
from .li_danner cimport PyLIDannerNeuron


cdef class PyNetwork:
    """ Python interface to Network ODE """

    def __cinit__(self, nneurons: int):
        """ C initialization for manual memory allocation """
        self.nneurons = nneurons
        self._network = <Network*>malloc(sizeof(Network))
        if self._network is NULL:
            raise MemoryError("Failed to allocate memory for Network")
        self.c_neurons = <Neuron **>malloc(nneurons * sizeof(Neuron *))
        # if self._network.neurons is NULL:
        #     raise MemoryError("Failed to allocate memory for neurons in Network")

        self.neurons = []
        cdef Neuron *c_neuron
        cdef PyLIDannerNeuron pyn

        for n in range(self.nneurons):
            self.neurons.append(PyLIDannerNeuron(f"{n}", 0))
            pyn = <PyLIDannerNeuron> self.neurons[n]
            c_neuron = (<Neuron*>pyn._neuron)
            self.c_neurons[n] = c_neuron

    def __dealloc__(self):
        """ Deallocate any manual memory as part of clean up """
        if self._network.neurons is not NULL:
            free(self._network.neurons)
        if self._network is not NULL:
            free(self._network)

    cpdef void test(self, data):
        cdef double[:] states = np.empty((2,))
        cdef double[:] dstates = np.empty((2,))
        cdef double[:] inputs = np.empty((10,))
        cdef double[:] weights = np.empty((10,))
        cdef double[:] noise = np.empty((10,))
        cdef double drive = 0.0
        cdef Neuron **neurons = self.c_neurons
        cdef unsigned int t, j
        for t in range(int(1000*1e3)):
            for j in range(self.nneurons):
                neurons[j][0].nstates
                neurons[j][0].ode_rhs_c(
                    0.0,
                    states,
                    dstates,
                    inputs,
                    weights,
                    noise,
                    drive,
                    neurons[j][0]
                )

    def step(self):
        """ Network step function """
        self._network.step()
