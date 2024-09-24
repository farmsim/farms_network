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

Leaky Integrator Node with persistent sodium channel based on Danner et.al.
"""

# from libc.stdio cimport printf
# from libc.stdlib cimport free, malloc
# from libc.string cimport strdup


# cdef void ode_rhs_c(
#     double time,
#     double[:] states,
#     double[:] dstates,
#     double[:] inputs,
#     double[:] weights,
#     double[:] noise,
#     double drive,
#     Node node
# ):
#     """ ODE """

#     # # Parameters
#     cdef LINapDannerNodeParameters params = (
#         <LINapDannerNodeParameters*> node.parameters
#     )[0]
#     # States
#     cdef double state_v = states[0]

#     # Drive inputs
#     cdef double d_e = params.m_e * drive + params.b_e # Excitatory drive
#     cdef double d_i = params.m_i * drive + params.b_i # Inhibitory drive

#     # Ileak
#     cdef double i_leak = params.g_leak * (state_v - params.e_leak)

#     # ISyn_Excitatory
#     cdef double i_syn_e = params.g_syn_e * d_e * (state_v - params.e_syn_e)

#     # ISyn_Inhibitory
#     cdef double i_syn_i = params.g_syn_i * d_i * (state_v - params.e_syn_i)

#     # Node inputs
#     cdef double _sum = 0.0
#     cdef unsigned int j
#     cdef double _node_out
#     cdef double _weight

#     # for j in range(node.ninputs):
#     #     _sum += node_inputs_eval_c(inputs[j], weights[j])

#     # # noise current
#     # cdef double i_noise = c_noise_current_update(
#     #     self.state_noise.c_get_value(), &(self.noise_params)
#     # )
#     # self.state_noise.c_set_value(i_noise)

#     # dV
#     cdef i_noise = 0.0
#     dstates[0] = (
#         -(i_leak + i_syn_e + i_syn_i + i_noise + _sum)/params.c_m
#     )


# cdef double output_c(double time, double[:] states, Node node):
#     """ Node output. """

#     cdef double state_v = states[0]
#     cdef double _n_out = 1000.0

#     # cdef LIDannerNodeParameters params = <LIDannerNodeParameters> node.parameters

#     # if state_v >= params.v_max:
#     #     _n_out = 1.0
#     # elif (params.v_thr <= state_v) and (state_v < params.v_max):
#     #     _n_out = (state_v - params.v_thr) / (params.v_max - params.v_thr)
#     # elif state_v < params.v_thr:
#     #     _n_out = 0.0
#     return _n_out


# cdef double node_inputs_eval_c(double _node_out, double _weight):
#     return 0.0


# cdef class PyLINapDannerNode(PyNode):
#     """ Python interface to Leaky Integrator Node with persistence sodium C-Structure """

#     def __cinit__(self):
#         # override defaults
#         self._node.model_type = strdup("LI_NAP_DANNER".encode('UTF-8'))
#         self._node.nstates = 2
#         self._node.nparameters = 13
#         # methods
#         self._node.ode_rhs_c = ode_rhs_c
#         self._node.output_c = output_c
#         # parameters
#         self._node.parameters = <LIDannerNodeParameters*>malloc(
#             sizeof(LIDannerNodeParameters)
#         )

#     def __dealloc__(self):
#         if self._node.name is not NULL:
#             free(self._node.parameters)
