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
    double usr_input,
    double[:] inputs,
    double[:] weights,
    double[:] noise,
    Node node
) noexcept:
    """ Node ODE """
    printf("Base implementation of ODE C function \n")


cdef double output(
    double time,
    double[:] states,
    double usr_input,
    double[:] inputs,
    double[:] weights,
    double[:] noise,
    Node node
) noexcept:
    """ Node output """
    printf("Base implementation of output C function \n")
    return 0.0


cdef class PyNode:
    """ Python interface to Node C-Structure"""

    def __cinit__(self):
        self.node = <Node*>malloc(sizeof(Node))
        if self.node is NULL:
            raise MemoryError("Failed to allocate memory for Node")
        self.node.name = NULL
        self.node.model_type = strdup("base".encode('UTF-8'))
        self.node.ode = ode
        self.node.output = output

    def __dealloc__(self):
        if self.node.name is not NULL:
            free(self.node.name)
        if self.node.model_type is not NULL:
            free(self.node.model_type)
        if self.node.parameters is not NULL:
            free(self.node.parameters)
        if self.node is not NULL:
            free(self.node)

    def __init__(self, name):
        self.name = name

    @classmethod
    def from_options(cls, node_options: NodeOptions):
        """ From node options """
        name: str = node_options.name
        ninputs: int = node_options.ninputs
        return cls(name)

    # Property methods for name
    @property
    def name(self):
        if self.node.name is NULL:
            return None
        return self.node.name.decode('UTF-8')

    @name.setter
    def name(self, value):
        if self.node.name is not NULL:
            free(self.node.name)
        self.node.name = strdup(value.encode('UTF-8'))

    # Property methods for model_type
    @property
    def model_type(self):
        if self.node.model_type is NULL:
            return None
        return self.node.model_type.decode('UTF-8')

    # Property methods for nstates
    @property
    def nstates(self):
        return self.node.nstates

    # Property methods for nparameters
    @property
    def nparameters(self):
        return self.node.nparameters

    # Property methods for ninputs
    @property
    def ninputs(self):
        return self.node.ninputs

    # Methods to wrap the ODE and output functions
    def ode(self, double time, states, dstates, usr_input, inputs, weights, noise):
        cdef double[:] c_states = states
        cdef double[:] c_dstates = dstates
        cdef double[:] c_inputs = inputs
        cdef double[:] c_weights = weights
        cdef double[:] c_noise = noise
        # Call the C function directly
        self.node.ode(
            time, c_states, c_dstates, usr_input, c_inputs, c_weights, c_noise, self.node[0]
        )

    def output(self, double time, states, usr_input, inputs, weights, noise):
        # Call the C function and return its result
        return self.node[0].output(time, states, usr_input, inputs, weights, noise, self.node[0])