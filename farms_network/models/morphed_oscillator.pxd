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


# from ..core.node cimport Node, PyNode


# cdef enum:

#     #STATES
#     NSTATES = 3
#     STATE_THETA = 0
#     STATE_R= 1
#     # Morphing function state
#     STATE_F = 2


# cdef packed struct MorphedOscillatorNodeParameters:

#     double f
#     double gamma
#     double mu
#     double zeta


# cdef:
#     void ode(
#         double time,
#         double* states,
#         double* derivatives,
#         double external_input,
#         double* network_outputs,
#         unsigned int* inputs,
#         double* weights,
#         double noise,
#         Node* node,
#         Edge** edges,
#     ) noexcept
#     double output(
#         double time,
#         double* states,
#         double external_input,
#         double* network_outputs,
#         unsigned int* inputs,
#         double* weights,
#         Node* node,
#         Edge** edges,
#     ) noexcept


# cdef class PyMorphedOscillatorNode(PyNode):
#     """ Python interface to MorphedOscillator Node C-Structure """

#     cdef:
#         MorphedOscillatorNodeParameters parameters
