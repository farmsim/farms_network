"""Leaky Integrator Neuron."""

import numpy as np

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
        self.dae = dae

        #: Initialize parameters
        self.tau = self.dae.add_c('tau_' + self.n_id,
                                  kwargs.get('tau', 0.1))
        self.bias = self.dae.add_c('bias_' + self.n_id,
                                   kwargs.get('bias', -2.75))
        #: pylint: disable=invalid-name
        self.D = self.dae.add_c('D_' + self.n_id,
                                kwargs.get('D', 1.0))

        #: Initialize states
        self.m = self.dae.add_x('m_' + self.n_id,
                                kwargs.get('x0', 0.0))
        #: External inputs
        self.ext_in = self.dae.add_u('ext_in_' + self.n_id)

        #: ODE RHS
        self.mdot = self.dae.add_y('mdot_' + self.n_id, 0.0)

        #: Neuron inputs
        self.neuron_inputs = []

    cdef real c_neuron_input_eval(self, NeuronInput n):
        """ Evaluate neuron input."""
        return n.neuron.c_output()*n.weight.value/self.tau.value

    def add_ode_input(self, neuron, **kwargs):
        """ Add relevant external inputs to the ode."""
        cdef Param weight = self.dae.add_p(
            'w_' + neuron.n_id + '_to_' + self.n_id,
            kwargs.get('weight'))

        cdef NeuronInput n = NeuronInput(neuron, weight)

        #: Append the neuron
        self.neuron_inputs.append(n)

    cdef void c_ode_rhs(self):
        """ Compute the ODE. Internal Setup Function."""
        #: Neuron inputs
        cdef real _sum = 0.0
        cdef unsigned int j
        cdef NeuronInput n

        for n in self.neuron_inputs:
            _sum += self.c_neuron_input_eval(n)

        self.mdot.value = (
            (-self.m.value + self.ext_in.value)/self.tau.value) + _sum

    cdef real c_output(self):
        """ Neuron output. """
        return 1. / (1. + np.exp(-self.D.value * (
            self.m.value + self.bias.value)))

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
