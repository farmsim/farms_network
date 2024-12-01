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

Rectified Linear Unit
"""


from ..core.node cimport Node, PyNode
from ..core.edge cimport Edge, PyEdge


cdef enum:
    #STATES
    NSTATES = 0


cdef packed struct ReLUNodeParameters:
    double gain
    double sign
    double offset


cdef:
    double output(
        double time,
        double* states,
        double external_input,
        double* network_outputs,
        unsigned int* inputs,
        double* weights,
        Node* node,
        Edge** edges,
    ) noexcept


cdef class PyReLUNode(PyNode):
    """ Python interface to ReLU Node C-Structure """

    cdef:
        ReLUNodeParameters parameters
