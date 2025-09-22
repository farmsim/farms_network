""" Oscillator model """


from ..core.node_cy cimport node_t, node_inputs_t, processed_inputs_t, NodeCy
from ..core.edge_cy cimport edge_t, EdgeCy


cdef enum:
    #STATES
    NSTATES = 3
    STATE_PHASE = 0
    STATE_AMPLITUDE= 1
    STATE_AMPLITUDE_0 = 2


cdef packed struct oscillator_params_t:

    double intrinsic_frequency     # Hz
    double nominal_amplitude       #
    double amplitude_rate          #


cdef packed struct oscillator_edge_params_t:

    double phase_difference        # radians


cdef void oscillator_input_tf(
    double time,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out
) noexcept


cdef void oscillator_ode(
    double time,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept


cdef double oscillator_output_tf(
    double time,
    const double* states,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept


cdef class OscillatorNodeCy(NodeCy):
    """ Python interface to Oscillator Node C-Structure """

    cdef:
        oscillator_params_t params


cdef class OscillatorEdgeCy(EdgeCy):
    """ Python interface to Oscillator Edge C-Structure """

    cdef:
        oscillator_edge_params_t params
