# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=True
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: overflowcheck=False
# cython: optimize.unpack_method_calls=True
# cython: np_pythran=False

"""Leaky Integrate and Fire Interneuron. Daun et """
from farms_container import Container
from libc.stdio cimport printf
import numpy as np
from libc.math cimport exp as cexp
from libc.math cimport cosh as ccosh
from libc.math cimport fabs as cfabs
cimport numpy as cnp


cdef class LIFDaunInterneuron(Neuron):
    """Leaky Integrate and Fire Interneuron.
    Based on Silvia Daun and Tbor's model.
    """

    def __init__(self, n_id, num_inputs, **kwargs):
        super(LIFDaunInterneuron, self).__init__('lif_daun_interneuron')

        self.n_id = n_id  #: Unique neuron identifier
        #: Get container
        container = Container.get_instance()

        #: Constants
        (_, self.g_nap) = container.neural.constants.add_parameter(
            'g_nap_' + self.n_id, kwargs.get('g_nap', 10.0))
        (_, self.e_nap) = container.neural.constants.add_parameter(
            'e_nap_' + self.n_id, kwargs.get('e_nap', 50.0))

        #: Parameters of h
        (_, self.v_h_h) = container.neural.constants.add_parameter(
            'v_h_h_' + self.n_id, kwargs.get('v_h_h', -30.0))
        (_, self.gamma_h) = container.neural.constants.add_parameter(
            'gamma_h_' + self.n_id, kwargs.get('gamma_h', 0.1667))

        #: Parameters of tau
        (_, self.v_t_h) = container.neural.constants.add_parameter(
            'v_t_h_' + self.n_id, kwargs.get('v_t_h', -30.0))
        (_, self.eps) = container.neural.constants.add_parameter(
            'eps_' + self.n_id, kwargs.get('eps', 0.0023))
        (_, self.gamma_t) = container.neural.constants.add_parameter(
            'gamma_t_' + self.n_id, kwargs.get('gamma_t', 0.0833))

        #: Parameters of m
        (_, self.v_h_m) = container.neural.constants.add_parameter(
            'v_h_m_' + self.n_id, kwargs.get('v_h_m', -37.0))
        (_, self.gamma_m) = container.neural.constants.add_parameter(
            'gamma_m_' + self.n_id, kwargs.get('gamma_m', -0.1667))

        #: Parameters of Ileak
        (_, self.g_leak) = container.neural.constants.add_parameter(
            'g_leak_' + self.n_id, kwargs.get('g_leak', 2.8))
        (_, self.e_leak) = container.neural.constants.add_parameter(
            'e_leak_' + self.n_id, kwargs.get('e_leak', -65.0))

        #: Other constants
        (_, self.c_m) = container.neural.constants.add_parameter(
            'c_m_' + self.n_id, kwargs.get('c_m', 0.9154))

        #: State Variables
        #: pylint: disable=invalid-name
        #: Membrane potential
        self.v = container.neural.states.add_parameter(
            'V_' + self.n_id, kwargs.get('v0', -60.0))[0]
        self.h = container.neural.states.add_parameter(
            'h_' + self.n_id, kwargs.get('h0', 0.0))[0]

        #: ODE
        self.vdot = container.neural.dstates.add_parameter(
            'vdot_' + self.n_id, 0.0)[0]
        self.hdot = container.neural.dstates.add_parameter(
            'hdot_' + self.n_id, 0.0)[0]

        #: External Input
        self.g_app = container.neural.inputs.add_parameter(
            'g_app_' + self.n_id, kwargs.get('g_app', 0.2))[0]
        self.e_app = container.neural.inputs.add_parameter(
            'e_app_' + self.n_id, kwargs.get('e_app', 0.0))[0]

        #: Output
        self.nout = container.neural.outputs.add_parameter(
            'nout_' + self.n_id, 0.0)[0]

        #: Neuron inputs
        self.num_inputs = num_inputs
        self.neuron_inputs = cnp.ndarray((num_inputs,),
                                         dtype=[('neuron_idx', 'i'),
                                                ('g_syn_idx', 'i'),
                                                ('e_syn_idx', 'i'),
                                                ('gamma_s_idx', 'i'),
                                                ('v_h_s_idx', 'i')])

    def add_ode_input(self,int idx, neuron, **kwargs):
        """ Add relevant external inputs to the ode.
        Parameters
        ----------
        neuron : <LIF_Danner>
            Neuron model from which the input is received.
        weight : <float>
            Strength of the synapse between the two neurons"""

        #: Create a struct to store the inputs and weights to the neuron
        cdef DaunInterNeuronInput n
        container = Container.get_instance()
        #: Get the neuron parameter
        neuron_idx = container.neural.outputs.get_parameter_index(
            'nout_'+neuron.n_id)

        g_syn = container.neural.parameters.add_parameter(
            'g_syn_' + self.n_id, kwargs.pop('g_syn', 0.0))[0]
        e_syn = container.neural.parameters.add_parameter(
            'e_syn_' + self.n_id, kwargs.pop('e_syn', 0.0))[0]
        gamma_s = container.neural.parameters.add_parameter(
            'gamma_s_' + self.n_id, kwargs.pop('gamma_s', 0.0))[0]
        v_h_s = container.neural.parameters.add_parameter(
            'v_h_s_' + self.n_id, kwargs.pop('v_h_s', 0.0))[0]

        #: Get neuron parameter indices
        g_syn_idx = container.neural.parameters.get_parameter_idx(
            'g_syn_' + self.n_id)
        e_syn_idx = container.neural.parameters.get_parameter_idx(
            'e_syn_' + self.n_id)
        gamma_s_idx = container.neural.parameters.get_parameter_idx(
            'gamma_s_' + self.n_id)
        v_h_s_idx = container.neural.parameters.get_parameter_idx(
            'v_h_s_' + self.n_id)

        #: Add the indices to the struct
        n.neuron_idx = neuron_idx
        n.g_syn_idx = g_syn_idx
        n.e_syn_idx = e_syn_idx
        n.gamma_s_idx = gamma_s_idx
        n.v_h_s_idx = v_h_s_idx

        #: Append the struct to the list
        self.neuron_inputs[idx] = n

    def output(_, self):
        """Neuron activation function.
        Parameters
        ----------
        m_potential: float
            Neuron membrane potential
        """
        return self.c_output()

    def ode_rhs(_, self, y, p):
        """ Python interface to the ode_rhs computation."""
        self.c_ode_rhs(y, p)

    #################### C-FUNCTIONS ####################

    cdef void c_ode_rhs(self, double[:] _y, double[:] _p) nogil:
        """ Compute the ODE. Internal Setup Function."""

        #: States
        cdef double _v = self.v.c_get_value()
        cdef double _h = self.h.c_get_value()

        #: tau_h(V)
        cdef double tau_h = 1./(self.eps*ccosh(self.gamma_t*(_v - self.v_t_h)))

        #: h_inf(V)
        cdef double h_inf = 1./(1. + cexp(self.gamma_h*(_v - self.v_h_h)))

        #: m_inf(V)
        cdef double m_inf = 1./(1. + cexp(self.gamma_m*(_v - self.v_h_m)))

        #: Inap
        #: pylint: disable=no-member
        cdef double i_nap = self.g_nap * m_inf * _h * (_v - self.e_nap)

        #: Ileak
        cdef double i_leak = self.g_leak * (_v - self.e_leak)

        #: Iapp
        cdef double i_app = self.g_app.c_get_value() * (
            _v - self.e_app.c_get_value())

        #: Neuron inputs
        cdef double _sum = 0.0
        cdef unsigned int j
        cdef double _neuron_out
        cdef double _g_syn
        cdef double _e_syn
        cdef double _gamma_s
        cdef double _v_h_s
        cdef DaunInterNeuronInput _neuron

        for j in range(self.num_inputs):
            _neuron = self.neuron_inputs[j]
            _neuron_out = _y[_neuron.neuron_idx]
            _g_syn = _p[_neuron.g_syn_idx]
            _e_syn = _p[_neuron.e_syn_idx]
            _gamma_s = _p[_neuron.gamma_s_idx]
            _v_h_s = _p[_neuron.v_h_s_idx]
            _sum += self.c_neuron_inputs_eval(
                _neuron_out, _g_syn, _e_syn, _gamma_s, _v_h_s)

        #: Slow inactivation
        self.hdot.c_set_value((h_inf - _h)/tau_h)

        #: dV
        self.vdot.c_set_value((-i_nap - i_leak - i_app - _sum)/self.c_m)

    cdef void c_output(self) nogil:
        """ Neuron output. """
        #: Set the neuron output
        self.nout.c_set_value(self.v.c_get_value())

    cdef double c_neuron_inputs_eval(
            self, double _neuron_out, double _g_syn, double _e_syn,
            double _gamma_s, double _v_h_s) nogil:
        """ Evaluate neuron inputs."""
        cdef double _v = self.v.c_get_value()

        cdef double _s_inf = 1./(1. + cexp(_gamma_s*(_neuron_out - _v_h_s)))

        return _g_syn*_s_inf*(_v - _e_syn)
