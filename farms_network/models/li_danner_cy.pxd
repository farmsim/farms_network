""" Leaky Integrator Node Based on Danner et.al. 2016 """


from ..core.node_cy cimport node_t, node_inputs_t, processed_inputs_t, NodeCy
from ..core.edge_cy cimport edge_t, EXCITATORY, INHIBITORY, CHOLINERGIC


cdef enum:
    #STATES
    NSTATES = 2
    STATE_V = 0
    STATE_A = 1


cdef packed struct li_danner_params_t:

    double c_m                     # pF
    double g_leak                  # nS
    double e_leak                  # mV
    double v_max                   # mV
    double v_thr                   # mV
    double g_syn_e                 # nS
    double g_syn_i                 # nS
    double e_syn_e                 # mV
    double e_syn_i                 # mV
    double tau_ch                  # ms


cdef void li_danner_input_tf(
    double time,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out
) noexcept


cdef void li_danner_ode(
    double time,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept


cdef double li_danner_output_tf(
    double time,
    const double* states,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept


cdef class LIDannerNodeCy(NodeCy):
    """ Python interface to LI Danner Node C-Structure """

    cdef:
        li_danner_params_t params
