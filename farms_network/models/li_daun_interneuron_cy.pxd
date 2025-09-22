""" Leaky Integrate and Fire InterNeuron Based on Daun et.al. """


from ..core.node_cy cimport node_t, node_inputs_t, processed_inputs_t, NodeCy
from ..core.edge_cy cimport edge_t, EXCITATORY, INHIBITORY, CHOLINERGIC


cdef enum:
    #STATES
    NSTATES = 2
    STATE_V = 0
    STATE_H = 1


cdef packed struct li_daun_params_t:

    double c_m
    double g_nap
    double e_nap
    double v_h_h
    double gamma_h
    double v_t_h
    double eps
    double gamma_t
    double v_h_m
    double gamma_m
    double g_leak
    double e_leak
