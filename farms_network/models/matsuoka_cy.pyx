""" Matsuoka Neuron model """

from libc.stdio cimport printf


cpdef enum STATE:
    #STATES
    nstates = NSTATES
    v = STATE_V
    w = STATE_W


cdef processed_inputs_t matsuoka_input_tf(
    double time,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
) noexcept:
    # Parameters
    cdef matsuoka_params_t params = (<matsuoka_params_t*> node[0].params)[0]

    # States
    cdef double state_v = states[<int>STATE.v]
    cdef double state_w = states[<int>STATE.w]

    cdef processed_inputs_t processed_inputs = {
        'generic': 0.0,
        'excitatory': 0.0,
        'inhibitory': 0.0,
        'cholinergic': 0.0,
        'phase_coupling': 0.0
    }

    cdef:
        double _sum = 0.0
        unsigned int j
        double _input, _weight

    for j in range(inputs.ninputs):
        _input = inputs.network_outputs[inputs.node_indices[j]]
        _weight = inputs.weights[j]


cdef void matsuoka_ode(
    double time,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept:
    pass


cdef double matsuoka_output_tf(
    double time,
    const double* states,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept:
    pass
