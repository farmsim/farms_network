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

Oscillator model
"""


from ..core.node cimport NodeCy, Node
from ..core.edge cimport EdgeCy, Edge


cdef enum:

    #STATES
    NSTATES = 3
    STATE_PHASE = 0
    STATE_AMPLITUDE= 1
    STATE_AMPLITUDE_0 = 2


cdef packed struct OscillatorNodeParameters:

    double intrinsic_frequency     # Hz
    double nominal_amplitude       #
    double amplitude_rate          #


cdef packed struct OscillatorEdgeParameters:

    double phase_difference        # radians


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


cdef class OscillatorNode(Node):
    """ Python interface to Oscillator Node C-Structure """

    cdef:
        OscillatorNodeParameters parameters


cdef class OscillatorEdge(Edge):
    """ Python interface to Oscillator Edge C-Structure """

    cdef:
        OscillatorEdgeParameters parameters
