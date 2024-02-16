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

Hopf Oscillator

[1]L. Righetti and A. J. Ijspeert, “Pattern generators with sensory
feedback for the control of quadruped locomotion,” in 2008 IEEE
International Conference on Robotics and Automation, May 2008,
pp. 819–824. doi: 10.1109/ROBOT.2008.4543306.

"""
from libc.math cimport exp
import numpy as np
cimport numpy as cnp

cdef class HopfOscillator(Neuron):

    def __init__(self, n_id, num_inputs, neural_container, **kwargs):
        """Initialize.

        Parameters
        ----------
        n_id: str
            Unique ID for the neuron in the network.
        """
        super(HopfOscillator, self).__init__('leaky', n_id)
        # Neuron ID
        self.n_id = n_id
        # Initialize parameters
        (_, self.mu) = neural_container.constants.add_parameter(
            'mu_' + self.n_id, kwargs.get('mu', 0.1))
        (_, self.omega) = neural_container.constants.add_parameter(
            'omega_' + self.n_id, kwargs.get('omega', 0.1))
        (_, self.alpha) = neural_container.constants.add_parameter(
            'alpha_' + self.n_id, kwargs.get('alpha', 1.0))
        (_, self.beta) = neural_container.constants.add_parameter(
            'beta_' + self.n_id, kwargs.get('beta', 1.0))

        # Initialize states
        self.x = neural_container.states.add_parameter(
            'x_' + self.n_id, kwargs.get('x0', 0.0))[0]
        self.y = neural_container.states.add_parameter(
            'y_' + self.n_id, kwargs.get('y0', 0.0))[0]
        # External inputs
        self.ext_in = neural_container.inputs.add_parameter(
            'ext_in_' + self.n_id)[0]

        # ODE RHS
        self.xdot = neural_container.dstates.add_parameter(
            'xdot_' + self.n_id, 0.0)[0]
        self.ydot = neural_container.dstates.add_parameter(
            'ydot_' + self.n_id, 0.0)[0]

        # Output
        self.nout = neural_container.outputs.add_parameter(
            'nout_' + self.n_id, 0.0)[0]

        # Neuron inputs
        self.neuron_inputs = cnp.ndarray((num_inputs,),
                                         dtype=[('neuron_idx', 'i'),
                                                ('weight_idx', 'i')])

        self.num_inputs = num_inputs

    def add_ode_input(self, int idx, neuron, neural_container, **kwargs):
        """ Add relevant external inputs to the ode."""
        # Create a struct to store the inputs and weights to the neuron
        cdef HopfOscillatorNeuronInput n
        # Get the neuron parameter
        neuron_idx = neural_container.outputs.get_parameter_index(
            'nout_'+neuron.n_id)

        # Add the weight parameter
        weight = neural_container.weights.add_parameter(
            'w_' + neuron.n_id + '_to_' + self.n_id, kwargs.get('weight', 0.0))
        weight_idx = neural_container.weights.get_parameter_index(
            'w_' + neuron.n_id + '_to_' + self.n_id)
        n.neuron_idx = neuron_idx
        n.weight_idx = weight_idx

        # Append the struct to the list
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
    cdef void c_ode_rhs(self, double[:] _y, double[:] _w, double[:] _p):
        """ Compute the ODE. Internal Setup Function."""
        # Neuron inputs
        cdef double _sum = 0.0
        cdef unsigned int j
        cdef double _neuron_out
        cdef double _weight

        for j in range(self.num_inputs):
            _neuron_out = _y[self.neuron_inputs[j].neuron_idx]
            _weight = _w[self.neuron_inputs[j].weight_idx]
            _sum += self.c_neuron_inputs_eval(_neuron_out, _weight)

        # sates
        cdef double x = self.x.c_get_value()
        cdef double y = self.y.c_get_value()
        cdef double mu = self.mu
        cdef double omega = self.omega
        self.xdot.c_set_value(
             self.alpha*(self.mu - (x**2 + y**2))*x - self.omega*y
        )
        self.ydot.c_set_value(
            self.beta*(self.mu - (x**2 + y**2))*y + self.omega*x + (
                self.ext_in.c_get_value() + _sum
            )
        )

    cdef void c_output(self):
        """ Neuron output. """
        self.nout.c_set_value(self.y.c_get_value())

    cdef double c_neuron_inputs_eval(self, double _neuron_out, double _weight):
        """ Evaluate neuron inputs."""
        return _neuron_out*_weight
