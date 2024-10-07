from .node cimport Node
from .data_cy cimport NetworkDataCy


cdef struct Network:

    # info
    unsigned long int nnodes
    unsigned long int nedges
    unsigned long int nstates

    # nodes list
    Node** nodes

cdef class PyNetwork:
    """ Python interface to Network ODE """

    cdef:
        Network *network
        public list pynodes
        public NetworkDataCy data

    # cpdef void step(self)
    cpdef void ode(self, double time, double[:] states)
