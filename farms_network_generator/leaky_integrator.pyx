"""Leaky Integrator Neuron."""
import numpy as np
cimport numpy as cnp
cimport cython

cdef class LeakyIntegrator(Neuron):

    def __init__(self, n_id, dae, num_inputs, **kwargs):
        """Initialize.

        Parameters
        ----------
        n_id: str
            Unique ID for the neuron in the network.
        """
        super(LeakyIntegrator, self).__init__('leaky')

        #: Neuron ID
        self.n_id = n_id

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
        self.mdot = dae.add_xdot('mdot_' + self.n_id, 0.0)

        #: Output
        self.neuron_out = dae.add_y('nout_' + self.n_id, 0.0)

        #: Neuron inputs
        self.neuron_inputs = cnp.ndarray((num_inputs,),
                                         dtype=NeuronInput)

        self.num_inputs = num_inputs

    def add_ode_input(self, idx, neuron_idx, weight_idx, **kwargs):
        """ Add relevant external inputs to the ode."""
        #: Create a struct to store the inputs and weights to the neuron
        cdef NeuronInput n = NeuronInput()
        n.neuron_idx = neuron_idx
        n.weight_idx = weight_idx
        #: Append the struct to the list
        self.neuron_inputs[idx] = n

    # cython: linetrace=True
    @cython.profile(True)
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void c_ode_rhs(self):
        """ Compute the ODE. Internal Setup Function."""
        #: Neuron inputs
        cdef real _sum = 0.0
        cdef unsigned int j

        for j in range(self.num_inputs):
            _sum += self.c_neuron_input_eval(self.neuron_inputs[j])

        self.mdot.c_set_value((
            (self.ext_in.c_get_value() - self.m.c_get_value())/self.tau.c_get_value()) + _sum)

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void c_output(self) nogil:
        """ Neuron output. """
        self.nout.set_value(1. / (1. + exp(-self.D.c_get_value() * (
            self.m.c_get_value() + self.bias.c_get_value()))))

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef real c_neuron_input_eval(self, NI n) nogil:
        """ Evaluate neuron input."""
        return n.neuron.c_output()*n.weight.c_get_value()/self.tau.c_get_value()

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
