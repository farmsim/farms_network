""" Relay model """

from libc.stdio cimport printf


cpdef enum STATE:
    #STATES
    nstates = NSTATES


cdef void relay_input_tf(
    double time,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out
) noexcept:

    out.generic = inputs.external_input


cdef void relay_ode(
    double time,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept:
    pass


cdef double relay_output_tf(
    double time,
    const double* states,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept:
    return input_vals.generic


cdef class RelayNodeCy(NodeCy):
    """ Python interface to Relay Node C-Structure """

    def __cinit__(self):
        # override default ode and out methods
        self._node.nstates = 0
        self._node.nparams = 0

        self._node.is_statefull = False
        self._node.input_tf = relay_input_tf
        self._node.output_tf = relay_output_tf

    def __init__(self, **kwargs):
        super().__init__()

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')
