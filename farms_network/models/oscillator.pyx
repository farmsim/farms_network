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


from libc.math cimport M_PI
from libc.math cimport sin as csin
from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strdup


cpdef enum STATE:

    #STATES
    nstates = NSTATES
    phase = STATE_PHASE
    amplitude = STATE_AMPLITUDE
    amplitude_0 = STATE_AMPLITUDE_0


cdef void ode(
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
) noexcept:
    """ ODE """
    # Parameters
    cdef OscillatorNodeParameters params = (<OscillatorNodeParameters*> c_node[0].parameters)[0]
    cdef OscillatorEdgeParameters edge_params

    # States
    cdef double state_phase = states[<int>STATE.phase]
    cdef double state_amplitude = states[<int>STATE.amplitude]
    cdef double state_amplitude_0 = states[<int>STATE.amplitude_0]

    cdef:
        double _sum = 0.0
        unsigned int j
        double _input, _weight

    cdef unsigned int ninputs = c_node.ninputs
    for j in range(ninputs):
        _input = network_outputs[inputs[j]]
        _weight = weights[j]
        edge_params = (<OscillatorEdgeParameters*> c_edges[j].parameters)[0]
        _sum += _weight*state_amplitude*csin(
            _input - state_phase - edge_params.phase_difference
        )

    # phidot : phase_dot
    derivatives[<int>STATE.phase] = 2*M_PI*params.intrinsic_frequency + _sum
    # ampdot
    derivatives[<int>STATE.amplitude] = state_amplitude_0
    derivatives[<int>STATE.amplitude_0] = params.amplitude_rate*(
        (params.amplitude_rate/4.0)*(params.nominal_amplitude - state_amplitude) - state_amplitude_0
    )


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
    return states[<int>STATE.phase]


cdef class OscillatorNode(Node):
    """ Python interface to Oscillator Node C-Structure """

    def __cinit__(self):
        self.c_node.model_type = strdup("OSCILLATOR".encode('UTF-8'))
        # override default ode and out methods
        self.c_node.is_statefull = True
        self.c_node.ode = ode
        self.c_node.output = output
        # parameters
        self.c_node.parameters = malloc(sizeof(OscillatorNodeParameters))
        if self.c_node.parameters is NULL:
            raise MemoryError("Failed to allocate memory for node parameters")

    def __init__(self, name: str, **kwargs):
        super().__init__(name)

        # Set node parameters
        cdef OscillatorNodeParameters* param = <OscillatorNodeParameters*>(self.c_node.parameters)
        param.intrinsic_frequency = kwargs.pop("intrinsic_frequency")
        param.nominal_amplitude = kwargs.pop("nominal_amplitude")
        param.amplitude_rate = kwargs.pop("amplitude_rate")
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @property
    def parameters(self):
        """ Parameters in the network """
        cdef OscillatorNodeParameters params = (<OscillatorNodeParameters*> self.c_node.parameters)[0]
        return params


cdef class OscillatorEdge(Edge):
    """ Python interface to Oscillator Edge C-Structure """

    def __cinit__(self):

        # parameters
        self.c_edge.parameters = malloc(sizeof(OscillatorEdgeParameters))
        if self.c_edge.parameters is NULL:
            raise MemoryError("Failed to allocate memory for edge parameters")

    def __init__(self, source: str, target: str, edge_type: str, **kwargs):
        super().__init__(source, target, edge_type)

        # Set edge parameters
        cdef OscillatorEdgeParameters* param = <OscillatorEdgeParameters*>(self.c_edge.parameters)
        param.phase_difference = kwargs.pop("phase_difference")

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @property
    def parameters(self):
        """ Parameters in the network """
        cdef OscillatorEdgeParameters params = (<OscillatorEdgeParameters*> self.c_edge.parameters)[0]
        return params
