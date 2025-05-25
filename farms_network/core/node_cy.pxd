""" Node Base Struture. """

from farms_network.core.edge_cy cimport edge_t


cdef struct node_t:
    # Generic parameters
    unsigned int nstates        # Number of state variables in the node.
    unsigned int ninputs        # Number of inputs to the node within the network
    unsigned int nparams        # Number of parameters in the node

    char* model                 # Type of the model (e.g., "empty").
    char* name                  # Unique name of the node.

    bint is_statefull           # Flag indicating whether the node is stateful. (ODE)

    # Parameters
    void* params                # Pointer to the parameters of the node.

    # Functions
    void ode(
        double time,
        double* states,
        double* derivatives,
        double external_input,
        double* network_outputs,
        unsigned int* inputs,
        double* weights,
        double noise,
        node_t* node,
        edge_t** edges,
    ) noexcept

    double output(
        double time,
        double* states,
        double external_input,
        double* network_outputs,
        unsigned int* inputs,
        double* weights,
        node_t* node,
        edge_t** edges,
    ) noexcept


cdef class NodeCy:
    """ Interface to Node C-Structure """
    cdef:
        node_t* _node
