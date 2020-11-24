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

"""
-----------------------------------------------------------------------
Copyright 2018-2020 Jonathan Arreguit, Shravan Tata Ramalingasetty
Copyright 2018 BioRobotics Laboratory, École polytechnique fédérale de Lausanne

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-----------------------------------------------------------------------

Morphed Oscillator model
"""
from libc.stdio cimport printf
import farms_pylog as pylog
from libc.math cimport exp
from libc.math cimport M_PI
from libc.math cimport sin as csin
import numpy as np
cimport numpy as cnp


cdef class MorphedOscillator(Neuron):

    def __init__(self, n_id, num_inputs, neural_container, **kwargs):
        """Initialize.

        Parameters
        ----------
        n_id: str
            Unique ID for the neuron in the network.
        """
        super(MorphedOscillator, self).__init__('leaky')

        #: Neuron ID
        self.n_id = n_id

        #: Initialize parameters
        (_, self.f) = neural_container.constants.add_parameter(
            'f_' + self.n_id, kwargs.get('f', 0.5))

        (_, self.gamma) = neural_container.constants.add_parameter(
            'g_' + self.n_id, kwargs.get('gamma', 100))

        (_, self.mu) = neural_container.constants.add_parameter(
            'mu_' + self.n_id, kwargs.get('mu', 1.0))

        (_, self.zeta) = neural_container.constants.add_parameter(
            'z_' + self.n_id, kwargs.get('zeta', 0.0))
        print(self.zeta)
        #: Initialize states
        self.theta = neural_container.states.add_parameter(
            'theta_' + self.n_id, kwargs.get('theta0', 0.0))[0]
        self.r = neural_container.states.add_parameter(
            'r_' + self.n_id, kwargs.get('r0', 0.0))[0]

        #: External inputs
        self.ext_in = neural_container.inputs.add_parameter(
            'ext_in_' + self.n_id)[0]

        #: Morphing function
        self.f_theta = neural_container.parameters.add_parameter(
            'f_theta_' + self.n_id, kwargs.get('f_theta0', 0.0))[0]
        self.fd_theta = neural_container.parameters.add_parameter(
            'fd_theta_' + self.n_id, kwargs.get('fd_theta0', 0.0))[0]

        #: ODE RHS
        self.theta_dot = neural_container.dstates.add_parameter(
            'theta_dot_' + self.n_id, 0.0)[0]
        self.r_dot = neural_container.dstates.add_parameter(
            'r_dot_' + self.n_id, 0.0)[0]

        #: Output
        self.nout = neural_container.outputs.add_parameter(
            'nout_' + self.n_id, 0.0)[0]

        #: Neuron inputs
        self.neuron_inputs = cnp.ndarray((num_inputs,),
                                         dtype=[('neuron_idx', 'i'),
                                                ('weight_idx', 'i'),
                                                ('theta_idx', 'i')])

        self.num_inputs = num_inputs

    def add_ode_input(self, int idx, neuron, neural_container, **kwargs):
        """ Add relevant external inputs to the ode."""
        #: Create a struct to store the inputs and weights to the neuron
        cdef MorphedOscillatorNeuronInput n
        #: Get the neuron parameter
        neuron_idx = neural_container.outputs.get_parameter_index(
            'nout_'+neuron.n_id)

        #: Add the weight parameter
        weight = neural_container.weights.add_parameter(
            'w_' + neuron.n_id + '_to_' + self.n_id,
            kwargs.get('weight', 0.0))[0]
        phi = neural_container.parameters.add_parameter(
            'phi_' + neuron.n_id + '_to_' + self.n_id,
            kwargs.get('phi', 0.0))[0]

        weight_idx = neural_container.weights.get_parameter_index(
            'w_' + neuron.n_id + '_to_' + self.n_id)
        phi_idx = neural_container.parameters.get_parameter_index(
            'phi_' + neuron.n_id + '_to_' + self.n_id)

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

    def ode_rhs(self, y, w, p):
        """ Python interface to the ode_rhs computation."""
        self.c_ode_rhs(y, w, p)

    #################### C-FUNCTIONS ####################
    cdef void c_ode_rhs(self, double[:] _y, double[:] _w, double[:] _p) nogil:
        """ Compute the ODE. Internal Setup Function."""

        #: Current state
        cdef double _theta = self.theta.c_get_value()
        cdef double _r = self.r.c_get_value()
        cdef double f_theta = self.f_theta.c_get_value()
        cdef double fd_theta = self.fd_theta.c_get_value()

        #: Neuron inputs
        cdef double _sum = 0.0
        cdef unsigned int j
        cdef double _neuron_out
        cdef double _weight
        cdef double _phi

        for j in range(self.num_inputs):
            _neuron_out = _y[self.neuron_inputs[j].neuron_idx]
            _weight = _w[self.neuron_inputs[j].weight_idx]
            _phi = _p[self.neuron_inputs[j].phi_idx]
            _sum += self.c_neuron_inputs_eval(
                _neuron_out, _weight, _theta, _phi)

        #: thetadot : theta_dot
        self.theta_dot.c_set_value(2*M_PI*self.f + _sum)

        #: rdot
        # cdef double r_dot_1 = 2*M_PI*self.f*_r*(fd_theta/f_theta)
        # cdef double r_dot_2 = _r*self.gamma*(self.mu - ((_r*_r)/(f_theta*f_theta)))
        # self.r_dot.c_set_value(r_dot_1 + r_dot_2 + self.zeta)

        cdef double r_dot_1 = fd_theta*self.theta_dot.c_get_value()
        cdef double r_dot_2 = self.gamma*(f_theta - _r)
        self.r_dot.c_set_value(r_dot_1 + r_dot_2 + self.zeta)

    cdef void c_output(self) nogil:
        """ Neuron output. """
        self.nout.c_set_value(self.theta.c_get_value())

    cdef double c_neuron_inputs_eval(
            self, double _neuron_out, double _weight, double _theta,
            double _phi) nogil:
        """ Evaluate neuron inputs."""
        return _weight*csin(_neuron_out - _theta - _phi)
