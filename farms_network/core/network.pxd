from .data_cy cimport NetworkDataCy
from .node cimport Node
from .edge cimport Edge


cdef struct Network:

    # info
    unsigned long int nnodes
    unsigned long int nedges
    unsigned long int nstates

    # nodes list
    Node** nodes

    # edges list
    Edge** edges


cdef class PyNetwork:
    """ Python interface to Network ODE """

    cdef:
        Network *network
        public list pynodes
        public list pyedges
        public NetworkDataCy data
        double[:] __tmp_node_outputs

    # cpdef void step(self)
    cpdef double[:] ode(self, double time, double[::1] states)
