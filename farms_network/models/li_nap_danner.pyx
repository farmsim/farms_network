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

from libc.math cimport cosh as ccosh
from libc.math cimport exp as cexp
from libc.math cimport fabs as cfabs
from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strdup

from ..core.options import LINaPDannerParameterOptions


cpdef enum STATE:

    #STATES
    nstates = NSTATES
    v = STATE_V
    h = STATE_H


cdef void ode(
    double time,
    double* states,
    double* derivatives,
    double external_input,
    double* network_outputs,
    unsigned int* inputs,
    double* weights,
    double noise,
    Node* node,
    Edge** edges,
) noexcept:
    """ ODE """
    cdef LINaPDannerNodeParameters params = (<LINaPDannerNodeParameters*> node[0].parameters)[0]

    # States
    cdef double state_v = states[<int>STATE.v]
    cdef double state_h = states[<int>STATE.h]

    # tau_h(V)
    cdef double tau_h = params.tau_0 + (params.tau_max - params.tau_0) / \
        ccosh((state_v - params.v1_2_t) / params.k_t)

    # h_inf(V)
    cdef double h_inf = 1./(1.0 + cexp((state_v - params.v1_2_h) / params.k_h))

    # m(V)
    cdef double m = 1./(1.0 + cexp((state_v - params.v1_2_m) / params.k_m))

    # Inap
    # pylint: disable=no-member
    cdef double i_nap = params.g_nap * m * state_h * (state_v - params.e_na)

    # Ileak
    cdef double i_leak = params.g_leak * (state_v - params.e_leak)

    # Neuron inputs
    cdef:
        double _sum = 0.0
        unsigned int j
        double _input, _weight

    cdef unsigned int ninputs = node.ninputs
    for j in range(ninputs):
        _input = network_outputs[inputs[j]]
        _weight = weights[j]
        if _weight >= 0.0:
            # Excitatory Synapse
            _sum += params.g_syn_e*cfabs(_weight)*_input*(state_v - params.e_syn_e)
        elif _weight < 0.0:
            # Inhibitory Synapse
            _sum += params.g_syn_i*cfabs(_weight)*_input*(state_v - params.e_syn_i)

    # noise current
    cdef double i_noise = noise

    # Slow inactivation
    derivatives[<int>STATE.h] = (h_inf - state_h) / tau_h

    # dV
    derivatives[<int>STATE.v] = -(i_nap + i_leak + i_noise + _sum)/params.c_m


cdef double output(
    double time,
    double* states,
    double external_input,
    double* network_outputs,
    unsigned int* inputs,
    double* weights,
    Node* node,
    Edge** edges,
) noexcept:
    """ Node output. """

    cdef LINaPDannerNodeParameters params = (<LINaPDannerNodeParameters*> node.parameters)[0]
    cdef double _n_out = 0.0
    cdef double state_v = states[<int>STATE.v]
    if state_v >= params.v_max:
        _n_out = 1.0
    elif (params.v_thr <= state_v) and (state_v < params.v_max):
        _n_out = (state_v - params.v_thr) / (params.v_max - params.v_thr)
    elif state_v < params.v_thr:
        _n_out = 0.0
    return _n_out


cdef class PyLINaPDannerNode(PyNode):
    """ Python interface to Leaky Integrator Node with persistence sodium C-Structure """

    def __cinit__(self):
        # override defaults
        self.node.model_type = strdup("LI_NAP_DANNER".encode('UTF-8'))
        # override default ode and out methods
        self.node.is_statefull = True
        self.node.ode = ode
        self.node.output = output
        # parameters
        self.node.parameters = malloc(sizeof(LINaPDannerNodeParameters))
        if self.node.parameters is NULL:
            raise MemoryError("Failed to allocate memory for node parameters")

    def __init__(self, name: str, **kwargs):
        super().__init__(name)

        # Set node parameters
        cdef LINaPDannerNodeParameters* param = <LINaPDannerNodeParameters*>(
            self.node.parameters
        )
        param.c_m = kwargs.pop("c_m")
        param.g_nap = kwargs.pop("g_nap")
        param.e_na = kwargs.pop("e_na")
        param.v1_2_m = kwargs.pop("v1_2_m")
        param.k_m = kwargs.pop("k_m")
        param.v1_2_h = kwargs.pop("v1_2_h")
        param.k_h = kwargs.pop("k_h")
        param.v1_2_t = kwargs.pop("v1_2_t")
        param.k_t = kwargs.pop("k_t")
        param.g_leak = kwargs.pop("g_leak")
        param.e_leak = kwargs.pop("e_leak")
        param.tau_0 = kwargs.pop("tau_0")
        param.tau_max = kwargs.pop("tau_max")
        param.v_max = kwargs.pop("v_max")
        param.v_thr = kwargs.pop("v_thr")
        param.g_syn_e = kwargs.pop("g_syn_e")
        param.g_syn_i = kwargs.pop("g_syn_i")
        param.e_syn_e = kwargs.pop("e_syn_e")
        param.e_syn_i = kwargs.pop("e_syn_i")
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @property
    def parameters(self):
        """ Parameters in the network """
        cdef LINaPDannerNodeParameters params = (
            <LINaPDannerNodeParameters*> self.node.parameters
        )[0]
        return params
