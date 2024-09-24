from .node cimport Node


cdef struct Network:
    unsigned int nstates
    Node* nodes
    void step()
    void ode_c()


# cdef void ode_c(
#     Node *nodes,
#     unsigned int iteration,
#     unsigned int nnodes,
#     double[:] dstates
# ):
#     """ Network step function """
#     cdef Node node
#     cdef NodeData node_data
#     cdef unsigned int j
#     cdef nnodes = sizeof(nodes)/sizeof(node)

#     # double[:, :] states
#     # double[:, :] dstates
#     # double[:, :] inputs
#     # double[:, :] weights
#     # double[:, :] noise

#     for j in range(nnodes):
#         node_data = network_data[j]
#         nodes[j].ode_rhs_c(
#             node_data.curr_state,
#             dstates,
#             inputs,
#             weights,
#             noise,
#             drive,
#             nodes[j]
#         )


cdef class PyNetwork:
    """ Python interface to Network ODE """

    cdef:
        Network *_network
        unsigned int nnodes
        list nodes
        Node **c_nodes

    cpdef void test(self, data)
