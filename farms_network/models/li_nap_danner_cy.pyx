from libc.math cimport cosh as ccosh
from libc.math cimport exp as cexp
from libc.math cimport fabs as cfabs
from libc.stdio cimport printf
from libc.string cimport strdup
import numpy as np

from farms_network.models import Models


cpdef enum STATE:
    #STATES
    nstates = NSTATES
    v = STATE_V
    h = STATE_H


cdef void li_nap_danner_input_tf(
    double time,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out
) noexcept:

    cdef li_nap_danner_params_t* params = (<li_nap_danner_params_t*> node[0].params)

    # States
    cdef double state_v = states[<int>STATE.v]
    cdef double state_h = states[<int>STATE.h]

    # Neuron inputs
    cdef:
        double _sum = 0.0
        unsigned int j
        double _input, _weight
        edge_t* _edge

    cdef unsigned int ninputs = inputs.ninputs
    for j in range(ninputs):
        _input = inputs.network_outputs[inputs.node_indices[j]]
        _weight = inputs.weights[j]
        _edge = edges[inputs.edge_indices[j]]
        if _edge.type == EXCITATORY:
            # Excitatory Synapse
            out.excitatory += params.g_syn_e*cfabs(_weight)*_input*(state_v - params.e_syn_e)
        elif _edge.type == INHIBITORY:
            # print(_input, _weight, inputs.source_indices[j], edges[j].type)
            # Inhibitory Synapse
            out.inhibitory += params.g_syn_i*cfabs(_weight)*_input*(state_v - params.e_syn_i)


cdef void li_nap_danner_ode(
    double time,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept:
    cdef li_nap_danner_params_t* params = (<li_nap_danner_params_t*> node[0].params)

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

    # noise current
    cdef double i_noise = noise

    # Slow inactivation
    derivatives[<int>STATE.h] = (h_inf - state_h) / tau_h

    # dV
    derivatives[<int>STATE.v] = -(
        i_nap + i_leak + i_noise + input_vals.excitatory + input_vals.inhibitory
    )/params.c_m


cdef double li_nap_danner_output_tf(
    double time,
    const double* states,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept:
    cdef li_nap_danner_params_t* params = (<li_nap_danner_params_t*> node[0].params)

    cdef double _n_out = 0.0
    cdef double state_v = states[<int>STATE.v]
    if state_v >= params.v_max:
        _n_out = 1.0
    elif (params.v_thr <= state_v) and (state_v < params.v_max):
        _n_out = (state_v - params.v_thr) / (params.v_max - params.v_thr)
    elif state_v < params.v_thr:
        _n_out = 0.0
    return _n_out


cdef class LINaPDannerNodeCy(NodeCy):
    """ Python interface to LI Danner NaP Node C-Structure """

    def __cinit__(self):
        self._node.nstates = 2
        self._node.nparams = 19

        self._node.is_statefull = True

        self._node.input_tf = li_nap_danner_input_tf
        self._node.ode = li_nap_danner_ode
        self._node.output_tf = li_nap_danner_output_tf
        # parameters
        self.params = li_nap_danner_params_t()
        self._node.params = <void*>&self.params
        if self._node.params is NULL:
            raise MemoryError("Failed to allocate memory for node parameters")

    def __init__(self, **kwargs):
        super().__init__()

        # Set node parameters
        self.params.c_m = kwargs.pop("c_m")
        self.params.g_nap = kwargs.pop("g_nap")
        self.params.e_na = kwargs.pop("e_na")
        self.params.v1_2_m = kwargs.pop("v1_2_m")
        self.params.k_m = kwargs.pop("k_m")
        self.params.v1_2_h = kwargs.pop("v1_2_h")
        self.params.k_h = kwargs.pop("k_h")
        self.params.v1_2_t = kwargs.pop("v1_2_t")
        self.params.k_t = kwargs.pop("k_t")
        self.params.g_leak = kwargs.pop("g_leak")
        self.params.e_leak = kwargs.pop("e_leak")
        self.params.tau_0 = kwargs.pop("tau_0")
        self.params.tau_max = kwargs.pop("tau_max")
        self.params.v_max = kwargs.pop("v_max")
        self.params.v_thr = kwargs.pop("v_thr")
        self.params.g_syn_e = kwargs.pop("g_syn_e")
        self.params.g_syn_i = kwargs.pop("g_syn_i")
        self.params.e_syn_e = kwargs.pop("e_syn_e")
        self.params.e_syn_i = kwargs.pop("e_syn_i")

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @property
    def parameters(self):
        """ Parameters in the network """
        cdef li_nap_danner_params_t params = (<li_nap_danner_params_t*> self._node.params)[0]
        return params
