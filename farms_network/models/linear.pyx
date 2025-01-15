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

Linear model
"""

from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strdup


cpdef enum STATE:

    #STATES
    nstates = NSTATES


cdef double output(
    double time,
    double* states,
    double external_input,
    double* network_outputs,
    unsigned int* inputs,
    double* weights,
    NodeCy* c_node,
    EdgeCy** c_edges,
) noexcept:
    """ Node output. """
    cdef LinearNodeParameters params = (<LinearNodeParameters*> c_node[0].parameters)[0]

    cdef:
        double _sum = 0.0
        unsigned int j
        double _input, _weight
    cdef unsigned int ninputs = c_node.ninputs
    _sum += external_input
    for j in range(ninputs):
        _input = network_outputs[inputs[j]]
        _weight = weights[j]
        _sum += _weight*_input

    return (params.slope*_sum + params.bias)


cdef class LinearNode(Node):
    """ Python interface to Linear Node C-Structure """

    def __cinit__(self):
        self.c_node.model_type = strdup("LINEAR".encode('UTF-8'))
        # override default ode and out methods
        self.c_node.is_statefull = False
        self.c_node.output = output
        # parameters
        self.c_node.parameters = malloc(sizeof(LinearNodeParameters))
        if self.c_node.parameters is NULL:
            raise MemoryError("Failed to allocate memory for node parameters")

    def __init__(self, name: str, **kwargs):
        super().__init__(name)

        # Set node parameters
        cdef LinearNodeParameters* param = <LinearNodeParameters*>(self.c_node.parameters)
        param.slope = kwargs.pop("slope")
        param.bias = kwargs.pop("bias")
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @property
    def parameters(self):
        """ Parameters in the network """
        cdef LinearNodeParameters params = (<LinearNodeParameters*> self.c_node.parameters)[0]
        return params
