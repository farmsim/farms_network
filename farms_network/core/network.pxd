cimport numpy as cnp

from ..numeric.integrators_cy cimport RK4Solver
from ..numeric.system cimport ODESystem
from .data_cy cimport NetworkDataCy, NodeDataCy
from .edge cimport Edge
from .node cimport Node

# Typedef for a function signature: ODE system f(t, y) -> dydt
ctypedef void (*ode_func)(void*, double, double[:], double[:])


cdef struct Network:

    # info
    unsigned long int nnodes
    unsigned long int nedges
    unsigned long int nstates

    # nodes list
    Node** nodes

    # edges list
    Edge** edges


cdef class PyNetwork(ODESystem):
    """ Python interface to Network ODE """

    cdef:
        Network *network
        public list pynodes
        public list pyedges
        public NetworkDataCy data
        double[:] __tmp_node_outputs

        Py_ssize_t iteration
        Py_ssize_t n_iterations
        double timestep

        public RK4Solver integrator

        list nodes_output_data

    # cpdef void step(self)
    cdef void evaluate(self, double time, double[:] states, double[:] derivatives) noexcept
    cpdef void step(self) noexcept

    cpdef void logging(self, Py_ssize_t) noexcept
