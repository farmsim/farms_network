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

from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strdup

from .options import NodeOptions


cdef void ode(
    double time,
    double[:] states,
    double[:] derivaties,
    double[:] inputs,
    double[:] weights,
    double[:] noise,
    Node node
) noexcept:
    """ Node ODE """
    printf("Base implementation of ODE C function \n")


cdef double output(double time, double[:] states, Node node):
    """ Node output """
    printf("Base implementation of node output C function \n")
    return 0.0


cdef class PyNode:
    """ Python interface to Node C-Structure"""

    def __cinit__(self):
        self._node = <Node*>malloc(sizeof(Node))
        if self._node is NULL:
            raise MemoryError("Failed to allocate memory for Node")
        self._node.name = NULL
        self._node.model_type = strdup("base".encode('UTF-8'))
        self._node.ode = ode
        self._node.output = output

    def __dealloc__(self):
        if self._node.name is not NULL:
            free(self._node.name)
        if self._node.model_type is not NULL:
            free(self._node.model_type)
        if self._node.parameters is not NULL:
            free(self._node.parameters)
        if self._node is not NULL:
            free(self._node)

    def __init__(self, name, ninputs):
        self.name = name
        self.ninputs = ninputs

    @classmethod
    def from_options(cls, node_options: NodeOptions):
        """ From node options """
        name: str = node_options.name
        ninputs: int = node_options.ninputs
        return cls(name, ninputs)

    # Property methods for name
    @property
    def name(self):
        if self._node.name is NULL:
            return None
        return self._node.name.decode('UTF-8')

    @name.setter
    def name(self, value):
        if self._node.name is not NULL:
            free(self._node.name)
        self._node.name = strdup(value.encode('UTF-8'))

    # Property methods for model_type
    @property
    def model_type(self):
        if self._node.model_type is NULL:
            return None
        return self._node.model_type.decode('UTF-8')

    # Property methods for nstates
    @property
    def nstates(self):
        return self._node.nstates

    # Property methods for nparameters
    @property
    def nparameters(self):
        return self._node.nparameters

    # Property methods for ninputs
    @property
    def ninputs(self):
        return self._node.ninputs

    @ninputs.setter
    def ninputs(self, value):
        self._node.ninputs = value

    # Methods to wrap the ODE and output functions
    def ode(self, double time, states, dstates, inputs, weights, noise):
        cdef double[:] c_states = states
        cdef double[:] c_dstates = dstates
        cdef double[:] c_inputs = inputs
        cdef double[:] c_weights = weights
        cdef double[:] c_noise = noise
        # Call the C function directly
        self._node.ode(
            time, c_states, c_dstates, c_inputs, c_weights, c_noise, self._node[0]
        )

    def output(self, double time, states):
        cdef double[:] c_states = states
        # Call the C function and return its result
        return self._node.output(time, c_states, self._node[0])
