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

Morphed Oscillator model
"""


# from libc.stdio cimport printf
# from libc.stdlib cimport malloc
# from libc.string cimport strdup


# cpdef enum STATE:

#     #STATES
#     nstates = NSTATES
#     theta = STATE_THETA
#     r = STATE_R
#     # Morphing function state
#     f = STATE_F


# cdef void ode(
#     double time,
#     double* states,
#     double* derivatives,
#     double external_input,
#     double* network_outputs,
#     unsigned int* inputs,
#     double* weights,
#     double noise,
#     Node* node,
#     Edge** edges,
# ) noexcept:
#     """ ODE """
#     # Parameters
#     cdef MorphedOscillatorNodeParameters params = (<MorphedOscillatorNodeParameters*> node[0].parameters)[0]

#     # States
#     cdef double state_x = states[<int>STATE.x]
#     cdef double state_y = states[<int>STATE.y]

#     cdef:
#         double _sum = 0.0
#         unsigned int j
#         double _input, _weight

#     cdef unsigned int ninputs = node.ninputs
#     for j in range(ninputs):
#         _input = network_outputs[inputs[j]]
#         _weight = weights[j]
#         _sum += (_weight*_input)

#     r_square = (state_x**2 + state_y**2)
#     # xdot : x_dot
#     derivatives[<int>STATE.x] = (
#         params.alpha*(params.mu - r_square)*state_x - params.omega*state_y
#     )
#     # ydot : y_dot
#     derivatives[<int>STATE.y] = (
#         params.beta*(params.mu - r_square)*state_y + params.omega*state_x + (_sum)
#     )


# cdef double output(
#     double time,
#     double* states,
#     double external_input,
#     double* network_outputs,
#     unsigned int* inputs,
#     double* weights,
#     Node* node,
#     Edge** edges,
# ) noexcept:
#     """ Node output. """
#     return states[<int>STATE.y]


# cdef class PyMorphedOscillatorNode(PyNode):
#     """ Python interface to MorphedOscillator Node C-Structure """

#     def __cinit__(self):
#         self.node.model_type = strdup("MORPHED_OSCILLATOR".encode('UTF-8'))
#         # override default ode and out methods
#         self.node.is_statefull = True
#         self.node.ode = ode
#         self.node.output = output
#         # parameters
#         self.node.parameters = malloc(sizeof(MorphedOscillatorNodeParameters))
#         if self.node.parameters is NULL:
#             raise MemoryError("Failed to allocate memory for node parameters")

#     def __init__(self, name: str, **kwargs):
#         super().__init__(name)

#         # Set node parameters
#         cdef MorphedOscillatorNodeParameters* params = <MorphedOscillatorNodeParameters*>(self.node.parameters)
#         params.f = kwargs.pop("f")
#         params.gamme = kwargs.pop("gamme")
#         params.mu = kwargs.pop("mu")
#         params.zeta = kwargs.pop("zeta")
#         if kwargs:
#             raise Exception(f'Unknown kwargs: {kwargs}')

#     @property
#     def parameters(self):
#         """ Parameters in the network """
#         cdef MorphedOscillatorNodeParameters params = (<MorphedOscillatorNodeParameters*> self.node.parameters)[0]
#         return params
