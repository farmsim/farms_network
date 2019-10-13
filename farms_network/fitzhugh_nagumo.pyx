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

"""Fitzhugh Nagumo model"""
from farms_container import Container
from libc.stdio cimport printf
import farms_pylog as pylog
import numpy as np
cimport numpy as cnp


cdef class FitzhughNagumo(Neuron):

    def __init__(self, n_id, num_inputs, **kwargs):
        """Initialize.

        Parameters
        ----------
        n_id: str
            Unique ID for the neuron in the network.
        """
        super(FitzhughNagumo, self).__init__('leaky')

        #: Neuron ID
        self.n_id = n_id
        #: Get container
        container = Container.get_instance()

        #: Initialize parameters

        (_, self.a) = container.neural.constants.add_parameter(
            'a_' + self.n_id, kwargs.get('a', 0.7))

        (_, self.b) = container.neural.constants.add_parameter(
            'b_' + self.n_id, kwargs.get('b', 0.8))

        (_, self.tau) = container.neural.constants.add_parameter(
            'tau_' + self.n_id, kwargs.get('tau', 1/0.08))

        (_, self.I) = container.neural.constants.add_parameter(
            'I_' + self.n_id, kwargs.get('I', 1))

        #: Initialize states
        self.V = container.neural.states.add_parameter(
            'V_' + self.n_id, kwargs.get('V0', 0.0))[0]
        self.w = container.neural.states.add_parameter(
            'w_' + self.n_id, kwargs.get('w0', 0.0))[0]
        
        #: External inputs
        self.ext_in = container.neural.inputs.add_parameter(
            'ext_in_' + self.n_id)[0]

        #: ODE RHS
        self.V_dot = container.neural.dstates.add_parameter(
            'V_dot_' + self.n_id, 0.0)[0]
        self.w_dot = container.neural.dstates.add_parameter(
            'w_dot_' + self.n_id, 0.0)[0]

        #: Output
        self.nout = container.neural.outputs.add_parameter(
            'nout_' + self.n_id, 0.0)[0]

        #: Neuron inputs
        self.neuron_inputs = cnp.ndarray((num_inputs,),
                                         dtype=[('neuron_idx', 'i'),
                                                ('weight_idx', 'i'),
                                                ('phi_idx', 'i')])

        self.num_inputs = num_inputs

    def add_ode_input(self, idx, neuron, **kwargs):
        """ Add relevant external inputs to the ode."""
        #: Create a struct to store the inputs and weights to the neuron
        cdef FNNeuronInput n = FNNeuronInput()
        container = Container.get_instance()
        #: Get the neuron parameter
        neuron_idx = container.neural.outputs.get_parameter_index(
            'nout_'+neuron.n_id)

        #: Add the weight parameter
        weight = container.neural.weights.add_parameter(
            'w_' + neuron.n_id + '_to_' + self.n_id,
            kwargs.get('weight', 0.0))[0]
        phi = container.neural.parameters.add_parameter(
            'phi_' + neuron.n_id + '_to_' + self.n_id,
            kwargs.get('phi', 0.0))[0]

        weight_idx = container.neural.weights.get_parameter_index(
            'w_' + neuron.n_id + '_to_' + self.n_id)
        phi_idx = container.neural.parameters.get_parameter_index(
            'phi_' + neuron.n_id + '_to_' + self.n_id)

        n.neuron_idx = neuron_idx
        n.weight_idx = weight_idx
        n.phi_idx = phi_idx
        cdef double x = self.a
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
    cdef void c_ode_rhs(self, double[:] _y, double[:] _p) nogil:
        """ Compute the ODE. Internal Setup Function."""

        #: Current state
        cdef double _V = self.V.c_get_value()
        cdef double _w = self.w.c_get_value()

        #: Neuron inputs
        cdef double _sum = 0.0
        cdef unsigned int j
        cdef double _neuron_out
        cdef double _weight
        cdef double _phi

        for j in range(self.num_inputs):
            _neuron_out = _y[self.neuron_inputs[j].neuron_idx]
            _weight = _p[self.neuron_inputs[j].weight_idx]
            _phi = _p[self.neuron_inputs[j].phi_idx]
            _sum += self.c_neuron_inputs_eval(_neuron_out,
                                              _weight, _phi, _V, _w)

        #: phidot : V_dot
        self.V_dot.c_set_value(_V - _V**3/3 - _w + self.I)

        #: wdot
        self.w_dot.c_set_value((1/self.tau)*(_V + self.a - self.b*_w))

    cdef void c_output(self) nogil:
        """ Neuron output. """
        self.nout.c_set_value(self.V.c_get_value())

    cdef double c_neuron_inputs_eval(
            self, double _neuron_out, double _weight, double _phi,
            double _V, double _w) nogil:
        """ Evaluate neuron inputs."""
        return _weight*(_neuron_out - _V - _phi)