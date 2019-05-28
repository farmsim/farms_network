"""Leaky Integrate and Fire Interneuron. Daun et """

from libc.stdio cimport printf
import numpy as np
from libc.math cimport exp as cexp
from libc.math cimport cosh as ccosh
from libc.math cimport fabs as cfabs
cimport numpy as cnp
cimport cython


cdef class HHDaunMotorneuron(Neuron):
    """Hodgkin Huxley Neuron Model
    Based on Silvia Daun and Tbor's model.
    """

    def __init__(self, n_id, dae, num_inputs, **kwargs):
        super(HHDaunMotorneuron, self).__init__('hh_daun_motorneuron')

        self.n_id = n_id  #: Unique neuron identifier

        #: Constants
        #: Neuron constants
        #: Parameters of INaP
        (self.g_nap, _) = dae.add_c('g_nap' + self.n_id,
                                    kwargs.get('g_nap', 10.0))
        (self.e_nap, _) = dae.add_c('e_nap' + self.n_id,
                                    kwargs.get('e_nap', 55.0))
        (self.am1_nap, _) = dae.add_c('am1_nap' + self.n_id,
                                      kwargs.get('am1_nap', 0.32))
        (self.am2_nap, _) = dae.add_c('am2_nap' + self.n_id,
                                      kwargs.get('am2_nap', -51.90))
        (self.am3_nap, _) = dae.add_c('am3_nap' + self.n_id,
                                      kwargs.get('am3_nap', 0.25))
        (self.bm1_nap, _) = dae.add_c('bm1_nap' + self.n_id,
                                      kwargs.get('bm1_nap', -0.280))
        (self.bm2_nap, _) = dae.add_c('bm2_nap' + self.n_id,
                                      kwargs.get('bm2_nap', -24.90))
        (self.bm3_nap, _) = dae.add_c('bm3_nap' + self.n_id,
                                      kwargs.get('bm3_nap', -0.2))
        (self.ah1_nap, _) = dae.add_c('ah1_nap' + self.n_id,
                                      kwargs.get('ah1_nap', 0.1280))
        (self.ah2_nap, _) = dae.add_c('ah2_nap' + self.n_id,
                                      kwargs.get('ah2_nap', -48.0))
        (self.ah3_nap, _) = dae.add_c('ah3_nap' + self.n_id,
                                      kwargs.get('ah3_nap', 0.0556))
        (self.bh1_nap, _) = dae.add_c('bh1_nap' + self.n_id,
                                      kwargs.get('bh1_nap', 4.0))
        (self.bh2_nap, _) = dae.add_c('bh2_nap' + self.n_id,
                                      kwargs.get('bh2_nap', -25.0))
        (self.bh3_nap, _) = dae.add_c('bh3_nap' + self.n_id,
                                      kwargs.get('bh3_nap', 0.20))

        #: Parameters of IK
        (self.g_k, _) = dae.add_c('g_k' + self.n_id,
                                  kwargs.get('g_k', 2.0))
        (self.e_k, _) = dae.add_c('e_k' + self.n_id,
                                  kwargs.get('e_k', -80.0))
        (self.am1_k, _) = dae.add_c('am1_k' + self.n_id,
                                    kwargs.get('am1_k', 0.0160))
        (self.am2_k, _) = dae.add_c('am2_k' + self.n_id,
                                    kwargs.get('am2_k', -29.90))
        (self.am3_k, _) = dae.add_c('am3_k' + self.n_id,
                                    kwargs.get('am3_k', 0.20))
        (self.bm1_k, _) = dae.add_c('bm1_k' + self.n_id,
                                    kwargs.get('bm1_k', 0.250))
        (self.bm2_k, _) = dae.add_c('bm2_k' + self.n_id,
                                    kwargs.get('bm2_k', -45.0))
        (self.bm3_k, _) = dae.add_c('bm3_k' + self.n_id,
                                    kwargs.get('bm3_k', 0.025))

        #: Parameters of Iq
        (self.g_q, _) = dae.add_c('g_q' + self.n_id,
                                  kwargs.get('g_q', 12.0))
        (self.e_q, _) = dae.add_c('e_q' + self.n_id,
                                  kwargs.get('e_q', -80.0))
        (self.gamma_q, _) = dae.add_c('gamma_q' + self.n_id,
                                      kwargs.get('gamma_q', -0.6))
        (self.r_q, _) = dae.add_c('r_q' + self.n_id,
                                  kwargs.get('r_q', 0.0005))
        (self.v_m_q, _) = dae.add_c('v_m_q' + self.n_id,
                                    kwargs.get('v_m_q', -30.0))

        #: Parameters of Ileak
        (self.g_leak, _) = dae.add_c('g_leak' + self.n_id,
                                     kwargs.get('g_leak', 0.8))
        (self.e_leak, _) = dae.add_c('e_leak' + self.n_id,
                                     kwargs.get('e_leak', -70.0))

        #: Parameters of Isyn
        (self.g_syn, _) = dae.add_c('g_syn' + self.n_id,
                                    kwargs.get('g_syn', 0.1))
        (self.e_syn, _) = dae.add_c('e_syn' + self.n_id,
                                    kwargs.get('e_syn', 0.0))
        (self.v_hs, _) = dae.add_c('v_hs' + self.n_id,
                                   kwargs.get('v_hs', -43.0))
        (self.gamma_s, _) = dae.add_c('gamma_s' + self.n_id,
                                      kwargs.get('gamma_s', -0.42))

        #: Other constants
        (self.c_m, _) = dae.add_c('c_m' + self.n_id,
                                  kwargs.get('c_m', 1.0))

        #: State Variables
        #: pylint: disable=invalid-name
        #: Membrane potential
        self.v = dae.add_x('V_' + self.n_id,
                           kwargs.get('v0', 0.0))
        self.m_na = dae.add_x(
            'm_na_' + self.n_id, kwargs.get('m_na0', 0.0))
        self.h_na = dae.add_x(
            'h_na_' + self.n_id, kwargs.get('h_na0', 0.0))
        self.m_k = dae.add_x('m_k_' + self.n_id,
                             kwargs.get('m_k0', 0.0))
        self.m_q = dae.add_x('m_q_' + self.n_id,
                             kwargs.get('m_q0', 0.0))

        #: ODE
        self.vdot = dae.add_xdot('vdot_' + self.n_id, 0.0)
        self.m_na_dot = dae.add_xdot('m_na_dot_' + self.n_id, 0.0)
        self.h_na_dot = dae.add_xdot('h_na_dot_' + self.n_id, 0.0)
        self.m_k_dot = dae.add_xdot('m_k_dot_' + self.n_id, 0.0)
        self.m_q_dot = dae.add_xdot('m_q_dot_' + self.n_id, 0.0)

        #: External Input
        self.g_app = dae.add_u('g_app_' + self.n_id,
                               kwargs.get('g_app', 0.19))
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
        cdef DaunMotorNeuronInput n = DaunMotorNeuronInput()

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
        cdef double _m_na = self.m_na.c_get_value()
        cdef double _h_na = self.h_na.c_get_value()
        cdef double _m_k = self.m_k.c_get_value()
        cdef double _m_q = self.m_q.c_get_value()

        #: alpha_m_Na(V)
        cdef double a_m_nap = (self.am1_nap * (self.am2_nap - _v)) / (
            cexp(self.am3_nap * (self.am2_nap - _v)) - 1)

        #: beta_m_Na(V)
        cdef double b_m_nap = (self.bm1_nap * (self.bm2_nap - _v)) / (
            cexp(self.bm3_nap * (self.bm2_nap - _v)) - 1)

        #: alpha_m_Na(V)
        cdef double a_h_nap = self.ah1_nap * cexp(
            self.ah3_nap * (self.ah2_nap - _v))

        #: beta_m_Na(V)
        cdef double b_h_nap = (self.bh1_nap) / (
            cexp(self.bh3_nap * (self.bh2_nap - _v)) + 1)

        #: Inap
        #: pylint: disable=no-member
        cdef double i_nap = self.g_nap * _m_na * _h_na * (
            _v - self.e_nap)

        #: alpha_m_K
        cdef double a_m_k = (self.am1_k * (self.am2_k - _v)) / (
            cexp(self.am3_k * (self.am2_k - _v)) - 1)

        #: beta_m_K
        cdef double b_m_k = self.bm1_k * cexp(self.bm3_k * (self.bm2_k - _v))

        #: Ik
        #: pylint: disable=no-member
        cdef double i_k = self.g_k * _m_k * (_v - self.e_k)

        #: m_q_inf
        cdef double m_q_inf = 1./(1 + cexp(self.gamma_q * (_v - self.v_m_q)))

        #: alpha_m_q
        cdef double a_m_q = m_q_inf * self.r_q

        #: beta_m_q
        cdef double b_m_q = (1 - m_q_inf) * self.r_q

        #: Ileak
        cdef double i_leak = self.g_leak * (_v - self.e_leak)

        #: Iapp
        cdef double i_app = self.g_app.c_get_value() * (
            _v - self.e_app.c_get_value())

        #: m_na_dot
        self.m_na_dot.c_set_value(a_m_nap*(1 - _m_na) - b_m_nap*_m_na)

        #: h_na_dot
        self.h_na_dot.c_set_value(a_h_nap*(1 - _h_na) - b_h_nap*_h_na)

        #: m_k_dot
        self.m_k_dot.c_set_value(a_m_k*(1 - _m_k) - b_m_k*_m_k)

        #: m_q_dot
        self.m_q_dot.c_set_value(a_m_q * (1 - _m_q) - b_m_q * _m_q)

        #: Iq
        #: pylint: disable=no-member
        cdef double i_q = self.g_q * self.m_q_dot.c_get_value() * (_v - self.e_q)

        #: Neuron inputs
        cdef double _sum = 0.0
        cdef unsigned int j
        cdef double _neuron_out
        cdef double _g_syn
        cdef double _e_syn
        cdef double _gamma_s
        cdef double _v_h_s
        cdef DaunMotorNeuronInput _neuron

        for j in range(self.num_inputs):
            _neuron = self.neuron_inputs[j]
            _neuron_out = _y[_neuron.neuron_idx]
            _g_syn = _p[_neuron.g_syn_idx]
            _e_syn = _p[_neuron.e_syn_idx]
            _gamma_s = _p[_neuron.gamma_s_idx]
            _v_h_s = _p[_neuron.v_h_s_idx]
            _sum += self.c_neuron_inputs_eval(
                _neuron_out, _g_syn, _e_syn, _gamma_s, _v_h_s)

        #: dV
        self.vdot.c_set_value((
            -i_nap - i_k - i_q - i_leak - i_app - _sum)/self.c_m)

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
