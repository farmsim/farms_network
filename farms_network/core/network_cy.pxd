cimport numpy as cnp

from ..numeric.integrators_cy cimport EulerMaruyamaSolver, RK4Solver
from ..numeric.system cimport ODESystem, SDESystem
from .data_cy cimport NetworkDataCy, NodeDataCy
from .edge cimport Edge, EdgeCy
from .node cimport Node, NodeCy


cdef struct NetworkStruct:

    # info
    unsigned long int nnodes
    unsigned long int nedges
    unsigned long int nstates

    # nodes list
    NodeCy** c_nodes

    # edges list
    EdgeCy** c_edges


cdef class NetworkCy(ODESystem):
    """ Python interface to Network ODE """

    cdef:
        NetworkStruct *c_network
        public list nodes
        public list edges
        public NetworkDataCy data
        double[:] __tmp_node_outputs

        unsigned int iteration
        unsigned int n_iterations
        unsigned int buffer_size
        double timestep

        public RK4Solver ode_integrator
        public EulerMaruyamaSolver sde_integrator

        SDESystem sde_system

        list nodes_output_data

    # cpdef void step(self)
    cdef void evaluate(self, double time, double[:] states, double[:] derivatives) noexcept
    cpdef void step(self)
