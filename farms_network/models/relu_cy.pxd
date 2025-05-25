""" Rectified Linear Unit """


from ..core.node_cy cimport node_t, NodeCy
from ..core.edge_cy cimport edge_t


cdef enum:
    #STATES
    NSTATES = 0


cdef packed struct relu_params_t:
    double gain
    double sign
    double offset


cdef:
    double output(
        double time,
        double* states,
        double external_input,
        double* network_outputs,
        unsigned int* inputs,
        double* weights,
        node_t* c_node,
        edge_t** c_edges,
    ) noexcept


cdef class ReLUNodeCy(NodeCy):
    """ Python interface to ReLU Node C-Structure """

    cdef:
        relu_params_t params
