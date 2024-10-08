from .data_cy cimport NetworkDataCy
from .node cimport Node


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
        double[:] __tmp_node_outputs

    # cpdef void step(self)
    cpdef double[:] ode(self, double time, double[::1] states)
