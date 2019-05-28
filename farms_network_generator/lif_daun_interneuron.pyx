"""Leaky Integrate and Fire Interneuron. Daun et """

from libc.stdio cimport printf
import numpy as np
from libc.math cimport exp as cexp
from libc.math cimport cosh as ccosh
from libc.math cimport fabs as cfabs
cimport numpy as cnp
cimport cython


cdef class LIFDaunInterneuron(Neuron):
    """Leaky Integrate and Fire Interneuron.
    Based on Silvia Daun and Tbor's model.
    """

    def __init__(self, n_id, dae, num_inputs, **kwargs):
        super(LIFDaunInterneuron, self).__init__('lif_daun_interneuron')

        self.n_id = n_id  #: Unique neuron identifier

        #: Constants
        (self.g_nap, _) = dae.add_c('g_nap_' + self.n_id,
                                    kwargs.get('g_nap', 10.0))
        (self.e_nap, _) = dae.add_c('e_nap_' + self.n_id,
                                    kwargs.get('e_nap', 50.0))

        #: Parameters of h
        (self.v_h_h, _) = dae.add_c('v_h_h_' + self.n_id,
                                    kwargs.get('v_h_h', -30.0))
        (self.gamma_h, _) = dae.add_c('gamma_h_' + self.n_id,
                                      kwargs.get('gamma_h', 0.1667))

        #: Parameters of tau
        (self.v_t_h, _) = dae.add_c('v_t_h_' + self.n_id,
                                    kwargs.get('v_t_h', -30.0))
        (self.eps, _) = dae.add_c('eps_' + self.n_id,
                                  kwargs.get('eps', 0.0023))
        (self.gamma_t, _) = dae.add_c('gamma_t_' + self.n_id,
                                      kwargs.get('gamma_t', 0.0833))

        #: Parameters of m
        (self.v_h_m, _) = dae.add_c('v_h_m_' + self.n_id,
                                    kwargs.get('v_h_m', -37.0))
        (self.gamma_m, _) = dae.add_c('gamma_m_' + self.n_id,
                                      kwargs.get('gamma_m', -0.1667))

        #: Parameters of Ileak
        (self.g_leak, _) = dae.add_c('g_leak_' + self.n_id,
                                     kwargs.get('g_leak', 2.8))
        (self.e_leak, _) = dae.add_c('e_leak_' + self.n_id,
                                     kwargs.get('e_leak', -65.0))

        #: Other constants
        (self.c_m, _) = dae.add_c('c_m_' + self.n_id,
                                  kwargs.get('c_m', 0.9154))

        #: State Variables
        #: pylint: disable=invalid-name
        #: Membrane potential
        self.v = dae.add_x('V_' + self.n_id,
                           kwargs.get('v0', -60.0))
        self.h = dae.add_x('h_' + self.n_id,
                           kwargs.get('h0', 0.0))

        #: ODE
        self.vdot = dae.add_xdot('vdot_' + self.n_id, 0.0)
        self.hdot = dae.add_xdot('hdot_' + self.n_id, 0.0)

        #: External Input
        self.g_app = dae.add_u('g_app_' + self.n_id,
                               kwargs.get('g_app', 0.2))
        self.e_app = dae.add_u('e_app_' + self.n_id,
                               kwargs.get('e_app', 0.0))

        #: Output
        self.nout = dae.add_y('nout_' + self.n_id, 0.0)

        #: Neuron inputs
        self.num_inputs = num_inputs
        self.neuron_inputs = cnp.ndarray((num_inputs,),
                                         dtype=[('neuron_idx', 'i'),
                                                ('g_syn_idx', 'i'),
                                                ('e_syn_idx', 'i'),
                                                ('gamma_s_idx', 'i'),
                                                ('v_h_s_idx', 'i')])

    def add_ode_input(self, idx, neuron, dae, **kwargs):
        """ Add relevant external inputs to the ode.
        Parameters
        ----------
        neuron : <LIF_Danner>
            Neuron model from which the input is received.
        weight : <float>
            Strength of the synapse between the two neurons"""

        #: Create a struct to store the inputs and weights to the neuron
        cdef DaunInterNeuronInput n = DaunInterNeuronInput()

        #: Get the neuron parameter
        neuron_idx = dae.y.get_idx('nout_'+neuron.n_id)

        g_syn = dae.add_p('g_syn_' + self.n_id,
                          kwargs.pop('g_syn', 0.0))
        e_syn = dae.add_p('e_syn_' + self.n_id,
                          kwargs.pop('e_syn', 0.0))
        gamma_s = dae.add_p('gamma_s_' + self.n_id,
                            kwargs.pop('gamma_s', 0.0))
        v_h_s = dae.add_p('v_h_s_' + self.n_id,
                          kwargs.pop('v_h_s', 0.0))

        #: Get neuron parameter indices
        g_syn_idx = dae.p.get_idx('g_syn_' + self.n_id)
        e_syn_idx = dae.p.get_idx('e_syn_' + self.n_id)
        gamma_s_idx = dae.p.get_idx('gamma_s_' + self.n_id)
        v_h_s_idx = dae.p.get_idx('v_h_s_' + self.n_id)

        #: Add the indices to the struct
        n.neuron_idx = neuron_idx
        n.g_syn_idx = g_syn_idx
        n.e_syn_idx = e_syn_idx
        n.gamma_s_idx = gamma_s_idx
        n.v_h_s_idx = v_h_s_idx

        #: Append the struct to the list
        self.neuron_inputs[idx] = n

    def output(self):
        """Neuron activation function.
        Parameters
        ----------
        m_potential: float
            Neuron membrane potential
        """
        return self.c_output()

    def ode_rhs(self, y, p):
        """ Python interface to the ode_rhs computation."""
        self.c_ode_rhs(y, p)

    #################### C-FUNCTIONS ####################

    @cython.profile(True)
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
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

    @cython.profile(True)
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void c_output(self) nogil:
        """ Neuron output. """
        #: Set the neuron output
        self.nout.c_set_value(self.v.c_get_value())

    @cython.profile(True)
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double c_neuron_inputs_eval(
            self, double _neuron_out, double _g_syn, double _e_syn,
            double _gamma_s, double _v_h_s) nogil:
        """ Evaluate neuron inputs."""
        cdef double _v = self.v.c_get_value()

        cdef double _s_inf = 1./(1. + cexp(_gamma_s*(_neuron_out - _v_h_s)))

        return _g_syn*_s_inf*(_v - _e_syn)
