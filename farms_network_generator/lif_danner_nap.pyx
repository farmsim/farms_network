"""Leaky Integrate and Fire Neuron Based on Danner et.al."""
import numpy as np
from libc.math cimport exp as cexp
from libc.math cimport cosh as ccosh
from libc.math cimport fabs as cfabs
cimport numpy as cnp
cimport cython


cdef class LIFDannerNap(Neuron):
    """Leaky Integrate and Fire Neuron Based on Danner et.al.
    """

    def __init__(self, n_id, dae, num_inputs, **kwargs):
        super(
            LIFDannerNap, self).__init__('lif_danner_nap')

        self.n_id = n_id  #: Unique neuron identifier

        #: Constants
        (self.c_m, _) = dae.add_c('c_m_' + self.n_id,
                                  kwargs.get('c_m', 10.0))  #: pF

        (self.g_nap, _) = dae.add_c(
            'g_nap_'+self.n_id, kwargs.get('g_nap', 4.5))  #: nS
        (self.e_na, _) = dae.add_c(
            'e_na_'+self.n_id, kwargs.get('e_na', 50.0))  #: mV

        (self.v1_2_m, _) = dae.add_c(
            'v1_2_m_' + self.n_id, kwargs.get('v1_2_m', -40.0))  #: mV
        (self.k_m, _) = dae.add_c(
            'k_m_' + self.n_id, kwargs.get('k_m', -6.0))  #: mV

        (self.v1_2_h, _) = dae.add_c(
            'v1_2_h_' + self.n_id, kwargs.get('v1_2_h', -45.0))  #: mV
        (self.k_h, _) = dae.add_c(
            'k_h_' + self.n_id, kwargs.get('k_h', 4.0))  #: mV

        (self.v1_2_t, _) = dae.add_c(
            'v1_2_t_' + self.n_id, kwargs.get('v1_2_t', -35.0))  #: mV
        (self.k_t, _) = dae.add_c(
            'k_t_' + self.n_id, kwargs.get('k_t', 15.0))  #: mV

        (self.g_leak, _) = dae.add_c(
            'g_leak_' + self.n_id, kwargs.get('g_leak', 4.5))  #: nS
        (self.e_leak, _) = dae.add_c(
            'e_leak_' + self.n_id, kwargs.get('e_leak', -62.5))  #: mV

        (self.tau_0, _) = dae.add_c(
            'tau_0_' + self.n_id, kwargs.get('tau_0', 80.0))  #: ms
        (self.tau_max, _) = dae.add_c(
            'tau_max_' + self.n_id, kwargs.get('tau_max', 160.0))  #: ms
        (self.tau_noise, _) = dae.add_c(
            'tau_noise_' + self.n_id, kwargs.get('tau_noise', 10.0))  #: ms

        (self.v_max, _) = dae.add_c(
            'v_max_' + self.n_id, kwargs.get('v_max', 0.0))  #: mV
        (self.v_thr, _) = dae.add_c(
            'v_thr_' + self.n_id, kwargs.get('v_thr', -50.0))  #: mV

        (self.g_syn_e, _) = dae.add_c(
            'g_syn_e_' + self.n_id, kwargs.get('g_syn_e', 10.0))  #: nS
        (self.g_syn_i, _) = dae.add_c(
            'g_syn_i_' + self.n_id, kwargs.get('g_syn_i', 10.0))  #: nS
        (self.e_syn_e, _) = dae.add_c(
            'e_syn_e_' + self.n_id, kwargs.get('e_syn_e', -10.0))  #: mV
        (self.e_syn_i, _) = dae.add_c(
            'e_syn_i_' + self.n_id, kwargs.get('e_syn_i', -75.0))  #: mV

        (self.m_e, _) = dae.add_c(
            'm_e_' + self.n_id, kwargs.pop('m_e', 0.0))  #: m_E,i
        (self.m_i, _) = dae.add_c(
            'm_i_' + self.n_id, kwargs.pop('m_i', 0.0))  #: m_I,i
        (self.b_e, _) = dae.add_c(
            'b_e_' + self.n_id, kwargs.pop('b_e', 0.0))  #: m_E,i
        (self.b_i, _) = dae.add_c(
            'b_i_' + self.n_id, kwargs.pop('b_i', 0.0))  #: m_I,i

        #: State Variables
        #: pylint: disable=invalid-name
        self.v = dae.add_x('V_' + self.n_id,
                           kwargs.get('v0'))  #: Membrane potential
        self.h = dae.add_x('h_' + self.n_id,
                           kwargs.get('h0'))
        # self.i_noise = dae.add_x('In_' + self.n_id)

        #: ODE
        self.vdot = dae.add_xdot('vdot_' + self.n_id, 0.0)
        self.hdot = dae.add_xdot('hdot_' + self.n_id, 0.0)

        #: Ouput
        self.nout = dae.add_y('nout_' + self.n_id, 0.0)

        #: External Input (BrainStem Drive)
        self.alpha = dae.add_u('alpha_' + self.n_id, 0.22)

        #: Neuron inputs
        self.num_inputs = num_inputs
        self.neuron_inputs = cnp.ndarray((num_inputs,),
                                         dtype=[('neuron_idx', 'i'),
                                                ('weight_idx', 'i')])

    def add_ode_input(self, idx, neuron, dae, **kwargs):
        """ Add relevant external inputs to the ode.
        Parameters
        ----------
        """

        #: Create a struct to store the inputs and weights to the neuron
        cdef DannerNapNeuronInput n = DannerNapNeuronInput()

        #: Get the neuron parameter
        neuron_idx = dae.y.get_idx('nout_'+neuron.n_id)

        #: Add the weight parameter
        weight = dae.add_p(
            'w_' + neuron.n_id + '_to_' + self.n_id, kwargs.get('weight'))
        weight_idx = dae.p.get_idx('w_' + neuron.n_id + '_to_' + self.n_id)
        n.neuron_idx = neuron_idx
        n.weight_idx = weight_idx

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
    cdef void c_ode_rhs(self, double[:] _y, double[:] _p):
        """ Compute the ODE. Internal Setup Function."""

        #: States
        cdef double _v = self.v.c_get_value()
        cdef double _h = self.h.c_get_value()

        #: Drive inputs
        cdef double d_e = self.m_e * self.alpha.c_get_value() + self.b_e
        cdef double d_i = self.m_i * self.alpha.c_get_value() + self.b_i

        #: tau_h(V)
        cdef double tau_h = self.tau_0 + (self.tau_max - self.tau_0) / \
            ccosh((_v - self.v1_2_t) / self.k_t)

        #: h_inf(V)
        cdef double h_inf = 1./(1.0 + cexp((_v - self.v1_2_h) / self.k_h))

        #: m(V)
        cdef double m = 1./(1.0 + cexp((_v - self.v1_2_m) / self.k_m))

        #: Inap
        #: pylint: disable=no-member
        cdef double i_nap = self.g_nap * m * _h * (_v - self.e_na)

        #: Ileak
        cdef double i_leak = self.g_leak * (_v - self.e_leak)

        #: ISyn_Excitatory
        cdef double i_syn_e = self.g_syn_e * d_e * (_v - self.e_syn_e)

        #: ISyn_Inhibitory
        cdef double i_syn_i = self.g_syn_i * d_i * (_v - self.e_syn_i)

        #: Neuron inputs
        cdef double _sum = 0.0
        cdef unsigned int j
        cdef double _neuron_out
        cdef double _weight

        for j in range(self.num_inputs):
            _neuron_out = _y[self.neuron_inputs[j].neuron_idx]
            _weight = _p[self.neuron_inputs[j].weight_idx]
            _sum += self.c_neuron_inputs_eval(_neuron_out, _weight)

        #: Slow inactivation
        self.hdot.c_set_value((h_inf - _h) / tau_h)

        #: dV
        self.vdot.c_set_value(
            -(i_nap + i_leak + i_syn_e + i_syn_i + _sum)/self.c_m)

    @cython.profile(True)
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void c_output(self):
        """ Neuron output. """
        cdef double _v = self.v.c_get_value()
        cdef double _n_out

        if _v >= self.v_max:
            _n_out = 1.
        elif self.v_thr <= _v < self.v_max:
            _n_out = (_v - self.v_thr) / (self.v_max - self.v_thr)
        else:
            _n_out = 0.0
        #: Set the neuron output
        self.nout.c_set_value(_n_out)

    @cython.profile(True)
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double c_neuron_inputs_eval(self, double _neuron_out, double _weight):
        """ Evaluate neuron inputs."""
        cdef double _v = self.v.c_get_value()

        if _weight >= 0.0:
            #: Excitatory Synapse
            return (
                self.g_syn_e*cfabs(_weight)*_neuron_out*(_v - self.e_syn_e))
        elif _weight < 0.0:
            #: Inhibitory Synapse
            return (
                self.g_syn_i*cfabs(_weight)*_neuron_out*(_v - self.e_syn_i))
