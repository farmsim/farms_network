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
    double* states,
    double* derivatives,
    double external_input,
    double* network_outputs,
    unsigned int* inputs,
    double* weights,
    double noise,
    Node* node,
    Edge** edges,
) noexcept:
    """ Node ODE """
    printf("Base implementation of ODE C function \n")


cdef double output(
    double time,
    double* states,
    double external_input,
    double* network_outputs,
    unsigned int* inputs,
    double* weights,
    Node* node,
    Edge** edges,
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
        self.node.nparameters = 0
        self.node.ninputs = 0

    def __dealloc__(self):
        if self.node.name is not NULL:
            free(self.node.name)
        if self.node.model_type is not NULL:
            free(self.node.model_type)
        if self.node.parameters is not NULL:
            free(self.node.parameters)
        if self.node is not NULL:
            free(self.node)

    def __init__(self, name: str, **kwargs):
        self.name = name

    @classmethod
    def from_options(cls, node_options: NodeOptions):
        """ From node options """
        name: str = node_options.name
        return cls(name, **node_options.parameters)

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

    # Property methods for ninputs
    @property
    def ninputs(self):
        return self.node.ninputs

    # Property methods for nparameters
    @property
    def nparameters(self):
        return self.node.nparameters

    @property
    def parameters(self):
        """Generic accessor for parameters."""
        if not self.node.parameters:
            raise ValueError("Node parameters are NULL")
        if self.node.nparameters == 0:
            raise ValueError("No parameters available")

        # The derived class should override this method to provide specific behavior
        raise NotImplementedError("Base class does not define parameter handling")

    # Methods to wrap the ODE and output functions
    def ode(
            self,
            double time,
            double[:] states,
            double[:] derivatives,
            double external_input,
            double[:] network_outputs,
            unsigned int[:] inputs,
            double[:] weights,
            double noise,
    ):
        cdef double* states_ptr = &states[0]
        cdef double* derivatives_ptr = &derivatives[0]
        cdef double* network_outputs_ptr = &network_outputs[0]
        cdef unsigned int* inputs_ptr = &inputs[0]
        cdef double* weights_ptr = &weights[0]

        cdef Edge** edges = NULL

        # Call the C function directly
        self.node.ode(
            time,
            states_ptr,
            derivatives_ptr,
            external_input,
            network_outputs_ptr,
            inputs_ptr,
            weights_ptr,
            noise,
            self.node,
            edges
        )

    def output(
            self,
            double time,
            double[:] states,
            double external_input,
            double[:] network_outputs,
            unsigned int[:] inputs,
            double[:] weights,
    ):
        # Call the C function and return its result
        cdef double* states_ptr = &states[0]
        cdef double* network_outputs_ptr = &network_outputs[0]
        cdef unsigned int* inputs_ptr = &inputs[0]
        cdef double* weights_ptr = &weights[0]
        cdef Edge** edges = NULL
        return self.node.output(
            time,
            states_ptr,
            external_input,
            network_outputs_ptr,
            inputs_ptr,
            weights_ptr,
            self.node,
            edges
        )
