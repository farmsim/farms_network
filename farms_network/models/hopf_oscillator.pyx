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

Hopf Oscillator

[1]L. Righetti and A. J. Ijspeert, “Pattern generators with sensory
feedback for the control of quadruped locomotion,” in 2008 IEEE
International Conference on Robotics and Automation, May 2008,
pp. 819–824. doi: 10.1109/ROBOT.2008.4543306.

"""

from libc.stdio cimport printf
from libc.stdlib cimport malloc
from libc.string cimport strdup


cpdef enum STATE:

    #STATES
    nstates = NSTATES
    x = STATE_X
    y = STATE_Y


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
    cdef HopfOscillatorNodeParameters params = (<HopfOscillatorNodeParameters*> c_node[0].parameters)[0]

    # States
    cdef double state_x = states[<int>STATE.x]
    cdef double state_y = states[<int>STATE.y]

    cdef:
        double _sum = 0.0
        unsigned int j
        double _input, _weight

    cdef unsigned int ninputs = c_node.ninputs
    for j in range(ninputs):
        _input = network_outputs[inputs[j]]
        _weight = weights[j]
        _sum += (_weight*_input)

    r_square = (state_x**2 + state_y**2)
    # xdot : x_dot
    derivatives[<int>STATE.x] = (
        params.alpha*(params.mu - r_square)*state_x - params.omega*state_y
    )
    # ydot : y_dot
    derivatives[<int>STATE.y] = (
        params.beta*(params.mu - r_square)*state_y + params.omega*state_x + (_sum)
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
    return states[<int>STATE.y]


cdef class HopfOscillatorNode(Node):
    """ Python interface to HopfOscillator Node C-Structure """

    def __cinit__(self):
        self.c_node.model_type = strdup("HOPF_OSCILLATOR".encode('UTF-8'))
        # override default ode and out methods
        self.c_node.is_statefull = True
        self.c_node.ode = ode
        self.c_node.output = output
        # parameters
        self.c_node.parameters = malloc(sizeof(HopfOscillatorNodeParameters))
        if self.c_node.parameters is NULL:
            raise MemoryError("Failed to allocate memory for node parameters")

    def __init__(self, name: str, **kwargs):
        super().__init__(name)

        # Set node parameters
        cdef HopfOscillatorNodeParameters* params = <HopfOscillatorNodeParameters*>(self.c_node.parameters)
        params.mu = kwargs.pop("mu")
        params.omega = kwargs.pop("omega")
        params.alpha = kwargs.pop("alpha")
        params.beta = kwargs.pop("beta")
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @property
    def parameters(self):
        """ Parameters in the network """
        cdef HopfOscillatorNodeParameters params = (<HopfOscillatorNodeParameters*> self.c_node.parameters)[0]
        return params
