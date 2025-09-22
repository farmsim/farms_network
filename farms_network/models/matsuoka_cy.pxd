""" Matsuoka Neuron model """


from ..core.node_cy cimport node_t, node_inputs_t, processed_inputs_t, NodeCy
from ..core.edge_cy cimport edge_t, EdgeCy


cdef enum:
    #STATES
    NSTATES = 2
    STATE_V = 0
    STATE_W= 1


cdef packed struct matsuoka_params_t:

    double c                    #
    double b                    #
    double tau                  #
    double T                    #
    double theta                #
    double nu                   #


cdef processed_inputs_t matsuoka_input_tf(
    double time,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
) noexcept


cdef void matsuoka_ode(
    double time,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept


cdef double matsuoka_output_tf(
    double time,
    const double* states,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept


cdef class MatsuokaNodeCy(NodeCy):
    """ Python interface to Matsuoka Node C-Structure """

    cdef:
        matsuoka_params_t params
