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
    v = STATE_V
    u = STATE_U


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
    """ Node ODE """
    # Parameters
    cdef IzhikevichNodeParameters params = (
        <IzhikevichNodeParameters*> c_node[0].parameters
    )[0]

    # States
    cdef double state_v = states[<int>STATE.v]
    cdef double state_u = states[<int>STATE.u]

    # Node inputs
    # cdef:
    #     double _sum = 0.0
    #     unsigned int j
    #     double _node_out, res, _input, _weight

    # cdef unsigned int ninputs = c_node.ninputs
    # for j in range(ninputs):
    #     _input = network_outputs[inputs[j]]
    #     _weight = weights[j]
    #     if _weight >= 0.0:
    #         # Excitatory Synapse
    #         _sum += params.g_syn_e*cfabs(_weight)*_input*(state_v - params.e_syn_e)
    #     elif _weight < 0.0:
    #         # Inhibitory Synapse
    #         _sum += params.g_syn_i*cfabs(_weight)*_input*(state_v - params.e_syn_i)

    # # dV
    # derivatives[<int>STATE.v] = 0.04*state_v**2 + 5.0*state_v + 140.0 - state_u + _sum
    # # dU
    # derivatives[<int>STATE.u] = params.a*(params.b*state_v - state_u)


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
    ...


cdef class IzhikevichNode(Node):
    """ Python interface to Izhikevich Node C-Structure """

    def __cinit__(self):
        self.c_node.model_type = strdup("IZHIKEVICH".encode('UTF-8'))
        # override default ode and out methods
        self.c_node.is_statefull = True
        self.c_node.output = output
        # parameters
        self.c_node.parameters = malloc(sizeof(IzhikevichNodeParameters))
        if self.c_node.parameters is NULL:
            raise MemoryError("Failed to allocate memory for node parameters")

    def __init__(self, name: str, **kwargs):
        super().__init__(name)

        # Set node parameters
        cdef IzhikevichNodeParameters* param = <IzhikevichNodeParameters*>(self.c_node.parameters)
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @property
    def parameters(self):
        """ Parameters in the network """
        cdef IzhikevichNodeParameters params = (<IzhikevichNodeParameters*> self.c_node.parameters)[0]
        return params
