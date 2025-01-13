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

Izhikevich neuron model based on Izhikevich et.al. 2003
"""

from ..core.node cimport Node, PyNode
from ..core.edge cimport Edge, PyEdge


cdef enum:
    #STATES
    NSTATES = 2
    STATE_V = 0
    STATE_U = 1



cdef packed struct IzhikevichNodeParameters:
    double a                    # recovery time scale
    double b                    # recovery sensitivity
    double c                    # after-spike reset
    double d                    # after-spike recovery reset


cdef:
    void ode(
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
    ) noexcept
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


cdef class PyIzhikevichNode(PyNode):
    """ Python interface to Izhikevich Node C-Structure """

    cdef:
        IzhikevichNodeParameters parameters
