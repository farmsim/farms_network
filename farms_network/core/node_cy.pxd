""" Node Base Struture. """

from farms_network.core.edge cimport EdgeCy


cdef struct node_t:
    # Generic parameters
    unsigned int nstates        # Number of state variables in the node.
    unsigned int nparameters    # Number of parameters for the node.
    unsigned int ninputs        # Number of inputs

    char* model                 # Type of the model (e.g., "empty").
    char* name                  # Unique name of the node.

    bint is_statefull              # Flag indicating whether the node is stateful. (ODE)

    # Parameters
    void* parameters            # Pointer to the parameters of the node.

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
        node_t* c_node,
        EdgeCy** c_edges,
    ) noexcept

    double output(
        double time,
        double* states,
        double external_input,
        double* network_outputs,
        unsigned int* inputs,
        double* weights,
        node_t* c_node,
        EdgeCy** c_edges,
    ) noexcept


cdef:
    void ode(
        double time,
        double* states,
        double* derivatives,
        double external_input,
        double* network_outputs,
        unsigned int* inputs,
        double* weights,
        double noise,
        node_t* c_node,
        EdgeCy** c_edges,
    ) noexcept
    double output(
        double time,
        double* states,
        double external_input,
        double* network_outputs,
        unsigned int* inputs,
        double* weights,
        node_t* c_node,
        EdgeCy** c_edges,
    ) noexcept


cdef class NodeCy:
    """ Interface to Node C-Structure """

    cdef:
        node_t* _node
