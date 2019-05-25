"""Leaky Integrator Neuron."""
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport exp


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
                                         dtype=[('neuron_idx', 'i'),
                                                ('weight_idx', 'i')])

        self.num_inputs = num_inputs

    def add_ode_input(self, idx, neuron_idx, weight_idx, **kwargs):
        """ Add relevant external inputs to the ode."""
        #: Create a struct to store the inputs and weights to the neuron
        cdef LeakyIntegratorNeuronInput n = LeakyIntegratorNeuronInput()
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
    cdef void c_ode_rhs(self, double[:] _y, double[:] _p) nogil:
        """ Compute the ODE. Internal Setup Function."""
        #: Neuron inputs
        cdef double _sum = 0.0
        cdef unsigned int j
        cdef double _neuron_out
        cdef double _weight
        
        for j in range(self.num_inputs):
             _neuron_out = _y[self.neuron_inputs[j].neuron_idx]
             _weight = _p[self.neuron_inputs[j].weight_idx]
             _sum += self.c_neuron_inputs_eval(_neuron_out, _weight)
        self.mdot.c_set_value((
            (self.ext_in.c_get_value() - self.m.c_get_value())/self.tau.c_get_value()) + _sum)

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void c_output(self) nogil:
        """ Neuron output. """
        self.nout.c_set_value(1. / (1. + exp(-self.D.c_get_value() * (
            self.m.c_get_value() + self.bias.c_get_value()))))

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double c_neuron_inputs_eval(self, double _neuron_out, double _weight) nogil:
        """ Evaluate neuron inputs."""
        return _neuron_out*_weight/self.tau.c_get_value()
