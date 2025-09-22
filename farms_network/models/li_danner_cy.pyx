""" Leaky Integrator Node based on Danner et.al. """

from libc.math cimport fabs as cfabs
from libc.stdio cimport printf
from libc.string cimport strdup

from farms_network.models import Models


cpdef enum STATE:
    #STATES
    nstates = NSTATES
    v = STATE_V
    a = STATE_A


cdef void li_danner_input_tf(
    double time,
    const double* states,
    const node_inputs_t inputs,
    const node_t* node,
    const edge_t** edges,
    processed_inputs_t* out
) noexcept:
    # Parameters
    cdef li_danner_params_t* params = (<li_danner_params_t*> node[0].params)

    # States
    cdef double state_v = states[<int>STATE.v]
    cdef double state_a = states[<int>STATE.a]

    # Node inputs
    cdef:
        double _sum = 0.0
        double _cholinergic_sum = 0.0
        unsigned int j
        double _node_out, res, _input, _weight
        edge_t* _edge

    cdef unsigned int ninputs = inputs.ninputs
    for j in range(ninputs):
        _input = inputs.network_outputs[inputs.node_indices[j]]
        _weight = inputs.weights[j]
        _edge = edges[inputs.edge_indices[j]]
        if _edge.type == EXCITATORY:
            # Excitatory Synapse
            out.excitatory += params.g_syn_e*cfabs(_weight)*_input*(state_v - params.e_syn_e)
        elif _edge.type == INHIBITORY:
            # Inhibitory Synapse
            out.inhibitory += params.g_syn_i*cfabs(_weight)*_input*(state_v - params.e_syn_i)
        elif _edge.type == CHOLINERGIC:
            out.cholinergic += cfabs(_weight)*_input


cdef void li_danner_ode(
    double time,
    const double* states,
    double* derivatives,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept:
    cdef li_danner_params_t* params = (<li_danner_params_t*> node[0].params)

    # States
    cdef double state_v = states[<int>STATE.v]
    cdef double state_a = states[<int>STATE.a]

    # Ileak
    cdef double i_leak = params.g_leak * (state_v - params.e_leak)

    # noise current
    cdef double i_noise = noise

    # da
    derivatives[<int>STATE.a] = (-state_a + input_vals.cholinergic)/params.tau_ch

    # dV
    derivatives[<int>STATE.v] = -(
        i_leak + i_noise + input_vals.excitatory + input_vals.inhibitory
    )/params.c_m


cdef double li_danner_output_tf(
    double time,
    const double* states,
    processed_inputs_t input_vals,
    double noise,
    const node_t* node,
) noexcept:
    cdef li_danner_params_t* params = (<li_danner_params_t*> node.params)

    cdef double _n_out = 0.0
    cdef double cholinergic_gain = 1.0
    cdef double state_v = states[<int>STATE.v]
    cdef double state_a = states[<int>STATE.a]

    if state_v >= params.v_max:
        _n_out = 1.0
    elif (params.v_thr <= state_v) and (state_v < params.v_max):
        _n_out = (state_v - params.v_thr) / (params.v_max - params.v_thr)
    elif state_v < params.v_thr:
        _n_out = 0.0
    if state_a > 0.0:
        cholinergic_gain = (1.0 + state_a)
    _n_out = min(cholinergic_gain*_n_out, 1.0)
    return _n_out


cdef class LIDannerNodeCy(NodeCy):
    """ Python interface to Leaky Integrator Node C-Structure """

    def __cinit__(self):
        self._node.nstates = 2
        self._node.nparams = 10

        self._node.is_statefull = True

        self._node.input_tf = li_danner_input_tf
        self._node.ode = li_danner_ode
        self._node.output_tf = li_danner_output_tf
        # parameters
        self.params = li_danner_params_t()
        self._node.params = <void*>&self.params
        if self._node.params is NULL:
            raise MemoryError("Failed to allocate memory for node parameters")

    def __init__(self, **kwargs):
        super().__init__()

        # Set node parameters
        self.params.c_m = kwargs.pop("c_m")
        self.params.g_leak = kwargs.pop("g_leak")
        self.params.e_leak = kwargs.pop("e_leak")
        self.params.v_max = kwargs.pop("v_max")
        self.params.v_thr = kwargs.pop("v_thr")
        self.params.g_syn_e = kwargs.pop("g_syn_e")
        self.params.g_syn_i = kwargs.pop("g_syn_i")
        self.params.e_syn_e = kwargs.pop("e_syn_e")
        self.params.e_syn_i = kwargs.pop("e_syn_i")
        self.params.tau_ch = kwargs.pop("tau_ch")

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @property
    def parameters(self):
        """ Parameters in the network """
        cdef li_danner_params_t params = (<li_danner_params_t*> self._node.params)[0]
        return params
