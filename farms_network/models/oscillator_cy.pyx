""" Oscillator model """


from libc.math cimport M_PI
from libc.math cimport sin as csin
from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strdup


cpdef enum STATE:

    #STATES
    nstates = NSTATES
    phase = STATE_PHASE
    amplitude = STATE_AMPLITUDE
    amplitude_0 = STATE_AMPLITUDE_0


cdef void oscillator_input_tf(
    double time,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out
) noexcept:

    # Parameters
    cdef oscillator_params_t* params = (<oscillator_params_t*> node[0].params)
    cdef oscillator_edge_params_t edge_params

    # States
    cdef double state_phase = states[<int>STATE.phase]
    cdef double state_amplitude = states[<int>STATE.amplitude]

    cdef:
        double _sum = 0.0
        unsigned int j
        double _input, _weight

    for j in range(inputs.ninputs):
        _input = inputs.network_outputs[inputs.node_indices[j]]
        _weight = inputs.weights[j]
        edge_params = (<oscillator_edge_params_t*> edges[inputs.edge_indices[j]].params)[0]
        out.generic += _weight*state_amplitude*csin(_input - state_phase - edge_params.phase_difference)


cdef void oscillator_ode(
    double time,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept:

    # Parameters
    cdef oscillator_params_t* params = (<oscillator_params_t*> node[0].params)
    cdef oscillator_edge_params_t edge_params

    # States
    cdef double state_phase = states[<int>STATE.phase]
    cdef double state_amplitude = states[<int>STATE.amplitude]
    cdef double state_amplitude_0 = states[<int>STATE.amplitude_0]

    cdef double input_val = input_vals.generic

    # phidot : phase_dot
    derivatives[<int>STATE.phase] = 2*M_PI*params.intrinsic_frequency + input_val
    # ampdot
    derivatives[<int>STATE.amplitude] = state_amplitude_0
    derivatives[<int>STATE.amplitude_0] = params.amplitude_rate*(
        (params.amplitude_rate/4.0)*(params.nominal_amplitude - state_amplitude) - state_amplitude_0
    )


cdef double oscillator_output_tf(
    double time,
    const double* states,
    processed_inputs_t input_val,
    double noise,
    const node_t* node,
) noexcept:
    return states[<int>STATE.phase]


cdef class OscillatorNodeCy(NodeCy):
    """ Python interface to Oscillator Node C-Structure """

    def __cinit__(self):
        # override default ode and out methods
        self._node.nstates = 3
        self._node.nparams = 3

        self._node.is_statefull = True
        self._node.input_tf = oscillator_input_tf
        self._node.ode = oscillator_ode
        self._node.output_tf = oscillator_output_tf
        # parameters
        self.params = oscillator_params_t()
        self._node.params = <void*>&self.params
        if self._node.params is NULL:
            raise MemoryError("Failed to allocate memory for node parameters")

    def __init__(self, **kwargs):
        super().__init__()

        # Set node parameters
        self.params.intrinsic_frequency = kwargs.pop("intrinsic_frequency")
        self.params.nominal_amplitude = kwargs.pop("nominal_amplitude")
        self.params.amplitude_rate = kwargs.pop("amplitude_rate")
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @property
    def parameters(self):
        """ Parameters in the network """
        cdef oscillator_params_t params = (<oscillator_params_t*> self._node.params)[0]
        return params


cdef class OscillatorEdgeCy(EdgeCy):
    """ Python interface to Oscillator Edge C-Structure """

    def __cinit__(self, edge_type: str, **kwargs):
        # parameters
        self.params = oscillator_edge_params_t()
        self._edge.params = <void*>&self.params
        self._edge.nparams = 1
        if self._edge.params is NULL:
            raise MemoryError("Failed to allocate memory for edge parameters")

    def __init__(self, edge_type: str, **kwargs):
        super().__init__(edge_type)

        # Set edge parameters
        self.params.phase_difference = kwargs.pop("phase_difference")
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @property
    def parameters(self):
        """ Parameters in the network """
        cdef oscillator_edge_params_t params = (<oscillator_edge_params_t*> self._edge.params)[0]
        return params
