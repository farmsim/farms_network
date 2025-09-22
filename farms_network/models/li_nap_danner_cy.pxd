"""
Leaky Integrator Node Based on Danner et.al. with Na and K channels
"""

from ..core.node_cy cimport node_t, node_inputs_t, processed_inputs_t, NodeCy
from ..core.edge_cy cimport edge_t, EXCITATORY, INHIBITORY, CHOLINERGIC


cdef enum:

    #STATES
    NSTATES = 2
    STATE_V = 0
    STATE_H = 1


cdef packed struct li_nap_danner_params_t:

    double c_m                  # pF
    double g_leak               # nS
    double e_leak               # mV
    double g_nap                # nS
    double e_na                 # mV
    double v1_2_m               # mV
    double k_m                  #
    double v1_2_h               # mV
    double k_h                  #
    double v1_2_t               # mV
    double k_t                  #
    double tau_0                # mS
    double tau_max              # mS
    double v_max                # mV
    double v_thr                # mV
    double g_syn_e                 # nS
    double g_syn_i                 # nS
    double e_syn_e                 # mV
    double e_syn_i                 # mV


cdef void li_nap_danner_input_tf(
    double time,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out
) noexcept


cdef void li_nap_danner_ode(
    double time,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept


cdef double li_nap_danner_output_tf(
    double time,
    const double* states,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept


cdef class LINaPDannerNodeCy(NodeCy):
    """ Python interface to LI Danner NaP Node C-Structure """

    cdef:
        li_nap_danner_params_t params
