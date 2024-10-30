cimport numpy as cnp

from .data_cy cimport NetworkDataCy, NodeDataCy
from .edge cimport Edge
from .node cimport Node


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

        Py_ssize_t iteration

    # cpdef void step(self)
    cpdef double[:] ode(self, double time, double[::1] states) noexcept
    # cdef void odeint(self, double time, double[:] states, double[:] derivatives) noexcept
    # cpdef void step(self) noexcept

    cpdef void logging(self, Py_ssize_t) noexcept
