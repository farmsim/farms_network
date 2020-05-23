# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=True
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: overflowcheck=False

"""Leaky Integrator Neuron."""
from libc.math cimport exp
import numpy as np
cimport numpy as cnp

cdef class LeakyIntegrator(Neuron):

    def __init__(self, n_id, num_inputs, neural_container, **kwargs):
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
        (_, self.tau) = neural_container.constants.add_parameter(
            'tau_' + self.n_id, kwargs.get('tau', 0.1))
        (_, self.bias) = neural_container.constants.add_parameter(
            'bias_' + self.n_id, kwargs.get('bias', -2.75))
        #: pylint: disable=invalid-name
        (_, self.D) = neural_container.constants.add_parameter(
            'D_' + self.n_id, kwargs.get('D', 1.0))

        #: Initialize states
        self.m = neural_container.states.add_parameter(
            'm_' + self.n_id, kwargs.get('x0', 0.0))[0]
        #: External inputs
        self.ext_in = neural_container.inputs.add_parameter(
            'ext_in_' + self.n_id)[0]

        #: ODE RHS
        self.mdot = neural_container.dstates.add_parameter(
            'mdot_' + self.n_id, 0.0)[0]

        #: Output
        self.nout = neural_container.outputs.add_parameter(
            'nout_' + self.n_id, 0.0)[0]

        #: Neuron inputs
        self.neuron_inputs = cnp.ndarray((num_inputs,),
                                         dtype=[('neuron_idx', 'i'),
                                                ('weight_idx', 'i')])

        self.num_inputs = num_inputs

    def add_ode_input(self, int idx, neuron, neural_container, **kwargs):
        """ Add relevant external inputs to the ode."""
        #: Create a struct to store the inputs and weights to the neuron
        cdef LeakyIntegratorNeuronInput n
        #: Get the neuron parameter
        neuron_idx = neural_container.outputs.get_parameter_index(
            'nout_'+neuron.n_id)

        #: Add the weight parameter
        weight = neural_container.weights.add_parameter(
            'w_' + neuron.n_id + '_to_' + self.n_id, kwargs.get('weight', 0.0))
        weight_idx = neural_container.weights.get_parameter_index(
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
        #: Neuron inputs
        cdef double _sum = 0.0
        cdef unsigned int j
        cdef double _neuron_out
        cdef double _weight

        for j in range(self.num_inputs):
            _neuron_out = _y[self.neuron_inputs[j].neuron_idx]
            _weight = _w[self.neuron_inputs[j].weight_idx]
            _sum += self.c_neuron_inputs_eval(_neuron_out, _weight)

        self.mdot.c_set_value((
            (-self.m.c_get_value() + _sum + self.ext_in.c_get_value())/self.tau)
        )

    cdef void c_output(self) nogil:
        """ Neuron output. """
        self.nout.c_set_value(1. / (1. + exp(-self.D * (
            self.m.c_get_value() + self.bias))))

    cdef double c_neuron_inputs_eval(self, double _neuron_out, double _weight) nogil:
        """ Evaluate neuron inputs."""
        return _neuron_out*_weight
