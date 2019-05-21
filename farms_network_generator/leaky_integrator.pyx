"""Leaky Integrator Neuron."""

import numpy as np
from libc.math cimport exp
cimport cython

cdef class NeuronInput(object):

    def __init__(self, neuron, weight):
        self.neuron = neuron
        self.weight = weight

cdef class LeakyIntegrator(object):

    def __init__(self, n_id, dae, **kwargs):
        """Initialize.

        Parameters
        ----------
        n_id: str
            Unique ID for the neuron in the network.
        """
        super(LeakyIntegrator, self).__init__()

        #: Neuron type
        self.neuron_type = 'leaky'

        #: Neuron ID
        self.n_id = n_id
        # self.dae = dae

        #: Initialize parameters
        self.tau = dae.add_c('tau_' + self.n_id,
                             kwargs.get('tau', 0.1))
        self.bias = dae.add_c('bias_' + self.n_id,
                              kwargs.get('bias', -2.75))
        #: pylint: disable=invalid-name
        self.D = dae.add_c('D_' + self.n_id,
                           kwargs.get('D', 1.0))

        #: Initialize states
        self.m = dae.add_x('m_' + self.n_id,
                           kwargs.get('x0', 0.0))
        #: External inputs
        self.ext_in = dae.add_u('ext_in_' + self.n_id)

        #: ODE RHS
        self.mdot = dae.add_y('mdot_' + self.n_id, 0.0)

        #: Neuron inputs
        self.neuron_inputs = np.ndarray((2,), dtype=NeuronInput)

    @cython.cdivision(False)
    cdef real c_neuron_input_eval(self, NeuronInput n):
        """ Evaluate neuron input."""
        return n.neuron.c_output()*n.weight.c_get_value()/self.tau.c_get_value()

    def add_ode_input(self, dae, neuron, idx, **kwargs):
        """ Add relevant external inputs to the ode."""
        cdef Param weight = dae.add_p(
            'w_' + neuron.n_id + '_to_' + self.n_id,
            kwargs.get('weight'))

        cdef NeuronInput n = NeuronInput(neuron, weight)

        #: Append the neuron
        print(self.neuron_inputs, idx)
        print(np.shape(self.neuron_inputs), idx)
        self.neuron_inputs[idx] = n

    # cython: linetrace=True
    @cython.profile(True)
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    @cython.cdivision(False)
    cdef void c_ode_rhs(self):
        """ Compute the ODE. Internal Setup Function."""
        #: Neuron inputs
        cdef real _sum = 0.0
        cdef unsigned int j
        cdef int n

        j = len(self.neuron_inputs)

        for n in range(2):
            _sum += self.c_neuron_input_eval(self.neuron_inputs[n])

        self.mdot.c_set_value((
            (self.ext_in.c_get_value() - self.m.c_get_value())/self.tau.c_get_value()) + _sum)

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    @cython.cdivision(False)
    cdef real c_output(self):
        """ Neuron output. """
        return 1. / (1. + exp(-self.D.c_get_value() * (
            self.m.c_get_value() + self.bias.c_get_value())))

    def output(self):
        """Neuron activation function.
        Parameters
        ----------
        m_potential: float
            Neuron membrane potential
        """
        return self.c_output()

    def ode_rhs(self):
        """ Python interface to the ode_rhs computation."""
        self.c_ode_rhs()
