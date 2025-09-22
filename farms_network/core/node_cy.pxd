""" Node Base Struture. """

from farms_network.core.edge_cy cimport edge_t


cdef struct node_inputs_t:
    double* network_outputs      # Network level outputs
    double* weights              # Network connection weights
    unsigned int* node_indices   # Which nodes provide input
    unsigned int* edge_indices   # Which edges provide input
    double external_input        # external input
    int ninputs                  # Number of inputs
    unsigned int node_index      # This node's index (for self-reference)


cdef struct processed_inputs_t:
    double generic
    double excitatory
    double inhibitory
    double cholinergic
    double phase_coupling


# Input transfer function
# Receives n-inputs and produces one output to be fed into ode/output_tf
cdef void base_input_tf(
    double time,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out
)


# ODE to compute the neural dynamics based on current state and inputs
cdef void base_ode(
    double time,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
)


# Output transfer function based on current state
cdef double base_output_tf(
    double time,
    const double* states,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
)


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
    void input_tf(
        double time,
        const double* states,
        const node_inputs_t inputs,
        const node_t* node,
        const edge_t** edges,
        processed_inputs_t* out
    ) noexcept
    void ode(
        double time,
        const double* states,
        double* derivatives,
        processed_inputs_t input_vals,
        double noise,
        const node_t* node,
    ) noexcept
    double output_tf(
        double time,
        const double* states,
        processed_inputs_t input_vals,
        double noise,
        const node_t* node,
    ) noexcept


cdef class NodeCy:
    """ Interface to Node C-Structure """
    cdef:
        node_t* _node
