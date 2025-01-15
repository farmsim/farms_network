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

Hopf-Oscillator model
"""


from ..core.node cimport NodeCy, Node
from ..core.edge cimport EdgeCy, Edge


cdef enum:

    #STATES
    NSTATES = 2
    STATE_X = 0
    STATE_Y= 1


cdef packed struct HopfOscillatorNodeParameters:

    double mu
    double omega
    double alpha
    double beta


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
        NodeCy* c_node,
        EdgeCy** c_edges,
    ) noexcept
    double output(
        double time,
        double* states,
        double external_input,
        double* network_outputs,
        unsigned int* inputs,
        double* weights,
        NodeCy* c_node,
        EdgeCy** c_edges,
    ) noexcept


cdef class HopfOscillatorNode(Node):
    """ Python interface to HopfOscillator Node C-Structure """

    cdef:
        HopfOscillatorNodeParameters parameters
