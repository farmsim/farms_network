""" Linear model """


from ..core.node_cy cimport node_t, node_inputs_t, processed_inputs_t, NodeCy
from ..core.edge_cy cimport edge_t


cdef enum:
    #STATES
    NSTATES = 0


cdef packed struct linear_params_t:
    double slope
    double bias


cdef void linear_input_tf(
    double time,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out
) noexcept


cdef void linear_ode(
    double time,
    const double* states,
    double* derivatives,
    processed_inputs_t input_val,
    double noise,
    const node_t* node,
) noexcept


cdef double linear_output_tf(
    double time,
    const double* states,
    processed_inputs_t input_val,
    double noise,
    const node_t* node,
) noexcept


cdef class LinearNodeCy(NodeCy):
    """ Python interface to Linear Node C-Structure """

    cdef:
        linear_params_t params
