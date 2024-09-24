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

Leaky Integrator Node based on Danner et.al.
"""

from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strdup


cdef void ode_rhs_c(
    double time,
    double[:] states,
    double[:] dstates,
    double[:] inputs,
    double[:] weights,
    double[:] noise,
    Node node
) noexcept:
    """ ODE """

    # # Parameters
    cdef LIDannerNodeParameters params = (
        <LIDannerNodeParameters*> node.parameters
    )[0]

    # States
    cdef double state_v = states[0]

    # Ileak
    cdef double i_leak = params.g_leak * (state_v - params.e_leak)

    # Node inputs
    cdef double _sum = 0.0
    cdef unsigned int j
    cdef double _node_out
    cdef double res

    cdef double _input
    cdef double _weight


    for j in range(10):
        _input = inputs[j]
        _weight = weights[j]
        _sum += node_inputs_eval_c(_input, _weight)

    # # noise current
    # cdef double i_noise = c_noise_current_update(
    #     self.state_noise.c_get_value(), &(self.noise_params)
    # )
    # self.state_noise.c_set_value(i_noise)

    # dV
    cdef double i_noise = 0.0
    dstates[0] = (
        -(i_leak + i_noise + _sum)/params.c_m
    )


cdef double output_c(double time, double[:] states, Node node):
    """ Node output. """

    cdef double state_v = states[0]
    cdef double _n_out = 1000.0

    # cdef LIDannerNodeParameters params = <LIDannerNodeParameters> node.parameters

    # if state_v >= params.v_max:
    #     _n_out = 1.0
    # elif (params.v_thr <= state_v) and (state_v < params.v_max):
    #     _n_out = (state_v - params.v_thr) / (params.v_max - params.v_thr)
    # elif state_v < params.v_thr:
    #     _n_out = 0.0
    return _n_out


cdef inline double node_inputs_eval_c(double _node_out, double _weight) noexcept:
    return 0.0


cdef class PyLIDannerNode(PyNode):
    """ Python interface to Leaky Integrator Node C-Structure """

    def __cinit__(self):
        self._node.model_type = strdup("LI_DANNER".encode('UTF-8'))
        # override default ode and out methods
        self._node.ode_rhs_c = ode_rhs_c
        self._node.output_c = output_c
        # parameters
        self._node.parameters = <LIDannerNodeParameters*>malloc(
            sizeof(LIDannerNodeParameters)
        )

    @classmethod
    def from_options(cls, node_options: NodeOptions):
        """ From node options """
        name: str = node_options.name
        ninputs: int = node_options.ninputs
        return cls(name, ninputs)
