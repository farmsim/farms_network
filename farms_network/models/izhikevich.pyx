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

Izhikevich model
"""

from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strdup


cpdef enum STATE:

    #STATES
    nstates = NSTATES


cdef void ode(
        double time,
        double* states,
        double* derivatives,
        double external_input,
        double* network_outputs,
        unsigned int* inputs,
        double* weights,
        Node* node,
        Edge** edges,
    ) noexcept:
    """ Node ODE """
    ...


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
    """ Node output. """
    ...


cdef class PyIzhikevichNode(PyNode):
    """ Python interface to Izhikevich Node C-Structure """

    def __cinit__(self):
        self.node.model_type = strdup("IZHIKEVICH".encode('UTF-8'))
        # override default ode and out methods
        self.node.is_statefull = True
        self.node.output = output
        # parameters
        self.node.parameters = malloc(sizeof(IzhikevichNodeParameters))
        if self.node.parameters is NULL:
            raise MemoryError("Failed to allocate memory for node parameters")

    def __init__(self, name: str, **kwargs):
        super().__init__(name)

        # Set node parameters
        cdef IzhikevichNodeParameters* param = <IzhikevichNodeParameters*>(self.node.parameters)
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @property
    def parameters(self):
        """ Parameters in the network """
        cdef IzhikevichNodeParameters params = (<IzhikevichNodeParameters*> self.node.parameters)[0]
        return params
