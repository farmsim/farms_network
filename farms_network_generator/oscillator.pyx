# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=True
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: overflowcheck=False

"""Oscillator model"""
from libc.math cimport exp
from libc.math cimport M_PI
from libc.math cimport sin as csin
import numpy as np
cimport numpy as cnp
cimport cython


cdef class Oscillator(Neuron):

    def __init__(self, n_id, dae, num_inputs, **kwargs):
        """Initialize.

        Parameters
        ----------
        n_id: str
            Unique ID for the neuron in the network.
        """
        super(Oscillator, self).__init__('leaky')

        #: Neuron ID
        self.n_id = n_id

        #: Initialize parameters
        (self.f, _) = dae.add_c('freq_' + self.n_id,
                                kwargs.get('f', 0.1))

        (self.R, _) = dae.add_c('R_' + self.n_id,
                                kwargs.get('R', 0.1))

        (self.a, _) = dae.add_c('a_' + self.n_id,
                                kwargs.get('a', 0.1))

        #: Initialize states
        self.phase = dae.add_x('phase_' + self.n_id,
                               kwargs.get('phase0', 0.0))
        self.amp = dae.add_x('amp_' + self.n_id,
                             kwargs.get('amp0', 0.0))
        #: External inputs
        self.ext_in = dae.add_u('ext_in_' + self.n_id)

        #: ODE RHS
        self.phase_dot = dae.add_xdot('phase_dot_' + self.n_id, 0.0)
        self.amp_dot = dae.add_xdot('amp_dot_' + self.n_id, 0.0)

        #: Output
        self.nout = dae.add_y('nout_' + self.n_id, 0.0)

        #: Neuron inputs
        self.neuron_inputs = cnp.ndarray((num_inputs,),
                                         dtype=[('neuron_idx', 'i'),
                                                ('weight_idx', 'i'),
                                                ('phi_idx', 'i')])

        self.num_inputs = num_inputs

    def add_ode_input(self, idx, neuron, dae, **kwargs):
        """ Add relevant external inputs to the ode."""
        #: Create a struct to store the inputs and weights to the neuron
        cdef OscillatorNeuronInput n = OscillatorNeuronInput()

        #: Get the neuron parameter
        neuron_idx = dae.y.get_idx('nout_'+neuron.n_id)

        #: Add the weight parameter
        weight = dae.add_p(
            'w_' + neuron.n_id + '_to_' + self.n_id, kwargs.get('weight'))
        phi = dae.add_p(
            'phi_' + neuron.n_id + '_to_' + self.n_id, kwargs.get('phi'))
        weight_idx = dae.p.get_idx('w_' + neuron.n_id + '_to_' + self.n_id)
        phi_idx = dae.p.get_idx('w_' + neuron.n_id + '_to_' + self.n_id)
        n.neuron_idx = neuron_idx
        n.weight_idx = weight_idx
        n.phi_idx = phi_idx

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
        cdef double _phase = self.phase.c_get_value()
        cdef double _amp = self.amp.c_get_value()

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
                                              _weight, _phi, _phase, _amp)

        #: phidot : phase_dot
        self.phase_dot.c_set_value(2*M_PI*self.f + _sum)

        #: ampdot
        self.amp_dot.c_set_value(self.a*(self.R - _amp))

    cdef void c_output(self) nogil:
        """ Neuron output. """
        self.nout.c_set_value(self.phase.c_get_value())

    cdef double c_neuron_inputs_eval(
            self, double _neuron_out, double _weight, double _phi,
            double _phase, double _amp) nogil:
        """ Evaluate neuron inputs."""
        return _weight*_amp**csin(_neuron_out - _phase - _phi)
