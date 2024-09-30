from .node cimport Node


cdef struct Network:

    # info
    unsigned long int nnodes
    unsigned long int nstates

    # nodes list
    Node* nodes

    # functions
    void ode(
        double time,
        unsigned int iteration,
        double[:] states,
        double[:] derivatives,
        Node **nodes,
    )


cdef void ode(
    double time,
    unsigned int iteration,
    double[:] states,
    double[:] derivatives,
    Node **nodes,
) noexcept
    # cdef Node __node
    # cdef NodeData node_data
    # cdef unsigned int j
    # cdef nnodes = sizeof(nodes)/sizeof(node)

    # double[:, :] states
    # double[:, :] dstates
    # double[:, :] inputs
    # double[:, :] weights
    # double[:, :] noise

    # for j in range(nnodes):
    #     node = node[j]
    #     node_data = network_data[j]
    #     if node.statefull:
    #         nstates = 2
    #         node.ode(t, state)
    #     nodes[j].step(
    #         time,
    #         node_data.curr_state,
    #         dstates,
    #         inputs,
    #         weights,
    #         noise,
    #         drive,
    #         nodes[j]
    #     )


cdef class PyNetwork:
    """ Python interface to Network ODE """

    cdef:
        Network *_network
        unsigned int nnodes
        list nodes
        Node **c_nodes

    # cpdef void step(self)
    cpdef void ode(self, double time, double[:] states)
