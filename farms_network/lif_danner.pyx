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

"""Leaky Integrate and Fire Neuron Based on Danner et.al."""
from farms_container import Container
from libc.stdio cimport printf
import numpy as np
from libc.math cimport exp as cexp
from libc.math cimport cosh as ccosh
from libc.math cimport fabs as cfabs
cimport numpy as cnp


cdef class LIFDanner(Neuron):
    """Leaky Integrate and Fire Neuron Based on Danner et.al.
    """

    def __init__(self, n_id, num_inputs, **kwargs):
        super(
            LIFDanner, self).__init__('lif_danner')

        self.n_id = n_id  #: Unique neuron identifier
        #: Get container
        container = Container.get_instance()
        
        #: Constants
        (_, self.c_m) = container.neural.constants.add_parameter(
            'c_m_' + self.n_id, kwargs.get('c_m', 10.0))  #: pF

        (_, self.g_leak) = container.neural.constants.add_parameter(
            'g_leak_' + self.n_id, kwargs.get('g_leak', 2.8))  # : nS
        (_, self.e_leak) = container.neural.constants.add_parameter(
            'e_leak_' + self.n_id, kwargs.get('e_leak', -60.0))  # : mV

        (_, self.tau_noise) = container.neural.constants.add_parameter(
            'tau_noise_' + self.n_id, kwargs.get('tau_noise', 10.0))  # : ms

        (_, self.v_max) = container.neural.constants.add_parameter(
            'v_max_' + self.n_id, kwargs.get('v_max', 0.0))  # : mV
        (_, self.v_thr) = container.neural.constants.add_parameter(
            'v_thr_' + self.n_id, kwargs.get('v_thr', -50.0))  # : mV

        (_, self.g_syn_e) = container.neural.constants.add_parameter(
            'g_syn_e_' + self.n_id, kwargs.get('g_syn_e', 10.0))  # : nS
        (_, self.g_syn_i) = container.neural.constants.add_parameter(
            'g_syn_i_' + self.n_id, kwargs.get('g_syn_i', 10.0))  # : nS
        (_, self.e_syn_e) = container.neural.constants.add_parameter(
            'e_syn_e_' + self.n_id, kwargs.get('e_syn_e', -10.0))  # : mV
        (_, self.e_syn_i) = container.neural.constants.add_parameter(
            'e_syn_i_' + self.n_id, kwargs.get('e_syn_i', -75.0))  # : mV

        (_, self.m_e) = container.neural.constants.add_parameter(
            'm_e_' + self.n_id, kwargs.pop('m_e', 0.0))  #: m_E,i
        (_, self.m_i) = container.neural.constants.add_parameter(
            'm_i_' + self.n_id, kwargs.pop('m_i', 0.0))  #: m_I,i
        (_, self.b_e) = container.neural.constants.add_parameter(
            'b_e_' + self.n_id, kwargs.pop('b_e', 0.0))  #: m_E,i
        (_, self.b_i) = container.neural.constants.add_parameter(
            'b_i_' + self.n_id, kwargs.pop('b_i', 0.0))  #: m_I,i

        #: State Variables
        #: pylint: disable=invalid-name
        self.v = container.neural.states.add_parameter(
            'V_' + self.n_id, kwargs.get('v0', -60.0))[0]  #: Membrane potential

        # self.i_noise = dae.add_x('In_' + self.n_id)

        #: ODE
        self.vdot = container.neural.dstates.add_parameter(
            'vdot_' + self.n_id, 0.0)[0]

        #: Output
        self.nout = container.neural.outputs.add_parameter(
            'nout_' + self.n_id, 0.0)[0]

        #: External Input (BrainStem Drive)
        self.alpha = container.neural.inputs.add_parameter(
            'alpha_' + self.n_id, 0.22)[0]

        #: Neuron inputs
        self.num_inputs = num_inputs
        self.neuron_inputs = cnp.ndarray((num_inputs,),
                                         dtype=[('neuron_idx', 'i'),
                                                ('weight_idx', 'i')])

    def add_ode_input(self, int idx, neuron, **kwargs):
        """ Add relevant external inputs to the ode.
        Parameters
        ----------
        """

        #: Create a struct to store the inputs and weights to the neuron
        cdef DannerNeuronInput n
        container = Container.get_instance()
        #: Get the neuron parameter
        neuron_idx = container.neural.outputs.get_parameter_index(
            'nout_'+neuron.n_id)

        #: Add the weight parameter
        weight = container.neural.weights.add_parameter(
            'w_' + neuron.n_id + '_to_' + self.n_id, kwargs.get('weight', 0.0))[0]
        weight_idx = container.neural.weights.get_parameter_index(
            'w_' + neuron.n_id + '_to_' + self.n_id)
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

    def ode_rhs(self, y, w, p):
        """ Python interface to the ode_rhs computation."""
        self.c_ode_rhs(y, w, p)

    #################### C-FUNCTIONS ####################

    cdef void c_ode_rhs(self, double[:] _y, double[:] _w, double[:] _p) nogil:
        """ Compute the ODE. Internal Setup Function."""

        #: States
        cdef double _v = self.v.c_get_value()

        #: Drive inputs
        cdef double d_e = self.m_e * self.alpha.c_get_value() + self.b_e
        cdef double d_i = self.m_i * self.alpha.c_get_value() + self.b_i

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
            _weight = _w[self.neuron_inputs[j].weight_idx]
            _sum += self.c_neuron_inputs_eval(_neuron_out, _weight)

        #: dV
        self.vdot.c_set_value(
            (-i_leak - i_syn_e - i_syn_i - _sum)/self.c_m)

    cdef void c_output(self) nogil:
        """ Neuron output. """
        cdef double _v = self.v.c_get_value()
        cdef double _n_out

        if _v >= self.v_max:
            _n_out = 1.
        elif self.v_thr <= _v < self.v_max:
            _n_out = (_v - self.v_thr) / (self.v_max - self.v_thr)
        elif _v < self.v_thr:
            _n_out = 0.0
        #: Set the neuron output
        self.nout.c_set_value(_n_out)

    cdef inline double c_neuron_inputs_eval(self, double _neuron_out, double _weight) nogil:
        """ Evaluate neuron inputs."""
        cdef double _v = self.v.c_get_value()

        if _weight >= 0.0:
            #: Excitatory Synapse
            return self.g_syn_e*cfabs(_weight)*_neuron_out*(_v - self.e_syn_e)
        elif _weight < 0.0:
            #: Inhibitory Synapse
            return self.g_syn_i*cfabs(_weight)*_neuron_out*(_v - self.e_syn_i)
