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
Matsuoka Neuron model
"""
from libc.stdio cimport printf
import farms_pylog as pylog
import numpy as np
cimport numpy as cnp


cdef class MatsuokaNeuron(Neuron):

    def __init__(self, n_id, num_inputs, neural_container, **kwargs):
        """Initialize.

        Parameters
        ----------
        n_id: str
            Unique ID for the neuron in the network.
        """
        super(MatsuokaNeuron, self).__init__('matsuoka_neuron')

        # Neuron ID
        self.n_id = n_id

        # Initialize parameters

        (_, self.c) = neural_container.constants.add_parameter(
            'c_' + self.n_id, kwargs.get('c', 1))

        (_, self.b) = neural_container.constants.add_parameter(
            'b_' + self.n_id, kwargs.get('b', 1))

        (_, self.tau) = neural_container.constants.add_parameter(
            'tau_' + self.n_id, kwargs.get('tau', 1))

        (_, self.T) = neural_container.constants.add_parameter(
            'T_' + self.n_id, kwargs.get('T', 12))

        (_, self.theta) = neural_container.constants.add_parameter(
            'theta_' + self.n_id, kwargs.get('theta', 0.0))

        (_, self.nu) = neural_container.constants.add_parameter(
            'nu' + self.n_id, kwargs.get('nu', 0.5))

        # Initialize states
        self.V = neural_container.states.add_parameter(
            'V_' + self.n_id, kwargs.get('V0', 0.0))[0]
        self.w = neural_container.states.add_parameter(
            'w_' + self.n_id, kwargs.get('w0', 0.5))[0]

        # External inputs
        self.ext_in = neural_container.inputs.add_parameter(
            'ext_in_' + self.n_id)[0]

        # ODE RHS
        self.V_dot = neural_container.dstates.add_parameter(
            'V_dot_' + self.n_id, 0.0)[0]
        self.w_dot = neural_container.dstates.add_parameter(
            'w_dot_' + self.n_id, 0.0)[0]

        # Output
        self.nout = neural_container.outputs.add_parameter(
            'nout_' + self.n_id, 0.0)[0]

        # Neuron inputs
        self.neuron_inputs = cnp.ndarray((num_inputs,),
                                         dtype=[('neuron_idx', 'i'),
                                                ('weight_idx', 'i'),
                                                ('phi_idx', 'i')])

        self.num_inputs = num_inputs

    def add_ode_input(self, int idx, neuron, neural_container, **kwargs):
        """ Add relevant external inputs to the ode."""
        # Create a struct to store the inputs and weights to the neuron
        cdef MatsuokaNeuronInput n
        # Get the neuron parameter
        neuron_idx = neural_container.outputs.get_parameter_index(
            'nout_'+neuron.n_id)

        # Add the weight parameter
        weight = neural_container.weights.add_parameter(
            'w_' + neuron.n_id + '_to_' + self.n_id,
            kwargs.get('weight', 2.5))[0]
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

        # Current state
        cdef double _V = self.V.c_get_value()
        cdef double _W = self.w.c_get_value()

        # Neuron inputs
        cdef double _sum = 0.0
        cdef unsigned int j
        cdef double _neuron_out
        cdef double _weight
        cdef double _phi

        for j in range(self.num_inputs):
            _neuron_out = _y[self.neuron_inputs[j].neuron_idx]
            _weight = _w[self.neuron_inputs[j].weight_idx]
            _phi = _p[self.neuron_inputs[j].phi_idx]
            _sum += self.c_neuron_inputs_eval(_neuron_out,
                                              _weight, _phi, _V, _W)

        # phidot : V_dot
        self.V_dot.c_set_value((1/self.tau)*(self.c - _V - _sum - self.b*_W))

        # wdot
        self.w_dot.c_set_value((1/self.T)*(-_W + self.nu*_V))

    cdef void c_output(self):
        """ Neuron output. """
        _V = self.V.c_get_value()
        if _V < 0:
            self.nout.c_set_value(max(-1, _V))
        else:
            self.nout.c_set_value(min(1, _V))

    cdef double c_neuron_inputs_eval(
            self, double _neuron_out, double _weight, double _phi,
            double _V, double _w):
        """ Evaluate neuron inputs."""
        return _weight*_neuron_out
