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

Leaky Integrator Neuron.
"""


from libc.math cimport exp as cexp
from libc.stdio cimport printf
from libc.stdlib cimport malloc
from libc.string cimport strdup

from ..core.options import LeakyIntegratorParameterOptions


cpdef enum STATE:

    #STATES
    nstates = NSTATES
    m = STATE_M


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
    cdef LeakyIntegratorNodeParameters params = (
        <LeakyIntegratorNodeParameters*> c_node[0].parameters
    )[0]

    # States
    cdef double state_m = states[<int>STATE.m]

    # Node inputs
    cdef:
        double _sum = 0.0
        unsigned int j
        double _node_out, res, _input, _weight

    cdef unsigned int ninputs = c_node.ninputs
    for j in range(ninputs):
        _input = network_outputs[inputs[j]]
        _weight = weights[j]
        _sum += _input*_weight

    # noise current
    cdef double i_noise = noise

    # dV
    derivatives[<int>STATE.m] = (-state_m + _sum + i_noise)/params.tau


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

    cdef LeakyIntegratorNodeParameters params = (
        <LeakyIntegratorNodeParameters*> c_node[0].parameters
    )[0]

    cdef double state_m = states[<int>STATE.m]
    cdef double _n_out = 1.0 / (1.0 + cexp(-params.D * (state_m + params.bias)))
    return _n_out


cdef class LeakyIntegratorNode(Node):
    """ Python interface to Leaky Integrator Node C-Structure """

    def __cinit__(self):
        self.c_node.model_type = strdup("LEAKY_INTEGRATOR".encode('UTF-8'))
        # override default ode and out methods
        self.c_node.is_statefull = True
        self.c_node.ode = ode
        self.c_node.output = output
        # parameters
        self.c_node.parameters = malloc(sizeof(LeakyIntegratorNodeParameters))
        if self.c_node.parameters is NULL:
            raise MemoryError("Failed to allocate memory for node parameters")

    def __init__(self, name: str, **kwargs):
        super().__init__(name)

        # Set node parameters
        cdef LeakyIntegratorNodeParameters* param = <LeakyIntegratorNodeParameters*>(self.c_node.parameters)
        param.tau = kwargs.pop("tau")
        param.bias = kwargs.pop("bias")
        param.D = kwargs.pop("D")
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @property
    def parameters(self):
        """ Parameters in the network """
        cdef LeakyIntegratorNodeParameters params = (<LeakyIntegratorNodeParameters*> self.c_node.parameters)[0]
        return params
