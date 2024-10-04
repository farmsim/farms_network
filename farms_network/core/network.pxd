from .node cimport Node
from .data_cy cimport NetworkDataCy


cdef struct Network:

    # info
    unsigned long int nnodes
    unsigned long int nedges
    unsigned long int nstates

    # nodes list
    Node **nodes

    # functions
    void ode(
        double time,
        unsigned int iteration,
        double[:] states,
        double[:] derivatives,
        Network network,
        NetworkDataCy data,
    )


cdef void ode(
    double time,
    unsigned int iteration,
    double[:] states,
    double[:] derivatives,
    Network network,
    NetworkDataCy data,
) noexcept


cdef class PyNetwork:
    """ Python interface to Network ODE """

    cdef:
        Network *network
        list nodes
        NetworkDataCy data

    # cpdef void step(self)
    cpdef void ode(self, double time, double[:] states)
