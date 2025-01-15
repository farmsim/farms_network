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

External model
"""

from libc.stdio cimport printf
from libc.stdlib cimport malloc
from libc.string cimport strdup


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
    return external_input


cdef class ExternalRelayNode(Node):
    """ Python interface to External Relay Node C-Structure """

    def __cinit__(self):
        self.c_node.model_type = strdup("EXTERNAL_RELAY".encode('UTF-8'))
        # override default ode and out methods
        self.c_node.is_statefull = False
        self.c_node.output = output

    def __init__(self, name: str, **kwargs):
        super().__init__(name)

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')
