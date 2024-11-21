cimport numpy as cnp

from ..numeric.integrators_cy cimport RK4Solver
from ..numeric.system cimport ODESystem
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


cdef class PyNetwork(ODESystem):
    """ Python interface to Network ODE """

    cdef:
        Network *network
        public list pynodes
        public list pyedges
        public NetworkDataCy data
        double[:] __tmp_node_outputs

        unsigned int iteration
        unsigned int n_iterations
        unsigned int buffer_size
        double timestep

        public RK4Solver integrator

        list nodes_output_data

    # cpdef void step(self)
    cdef void evaluate(self, double time, double[:] states, double[:] derivatives) noexcept
    cpdef void step(self) noexcept

    @staticmethod
    cdef void logging(unsigned int iteration, NetworkDataCy data, Network* network) noexcept
