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

Sensory afferent neurons.
"""

cimport numpy as cnp

from farms_network.neuron import Neuron
from libc.stdio cimport printf

cdef class ReLUNeuron(Neuron):
    """ Rectified Linear Unit neurons connecting """

    def __init__(self, n_id, num_inputs, neural_container, **kwargs):
        """Initialize.
        Parameters
        ----------
        n_id: str
            Unique ID for the neuron in the network.
        """
        super().__init__('relu')

        # Neuron ID
        self.n_id = n_id

        # Initialize parameters
        self.gain = neural_container.parameters.add_parameter(
            'gain_' + self.n_id, kwargs.get('gain', 1.0))[0]

        self.sign = neural_container.parameters.add_parameter(
            'sign_' + self.n_id, kwargs.get('sign', 1.0))[0]

        # assert abs(self.sign.value) != 1.0, "ReLU sign parameter should be 1.0"

        self.offset = neural_container.parameters.add_parameter(
            'offset_' + self.n_id, kwargs.get('offset', 0.0))[0]

        self.ext_inp = neural_container.inputs.add_parameter(
            'ext_' + self.n_id, kwargs.get('init', 0.0))[0]

        # Output
        self.nout = neural_container.outputs.add_parameter(
            'nout_' + self.n_id, 0.0)[0]

        # Neuron inputs
        self.num_inputs = num_inputs
        self.neuron_inputs = cnp.ndarray((num_inputs,),
                                         dtype=[('neuron_idx', 'i'),
                                                ('weight_idx', 'i')])

    def reset_sensory_param(self, param):
        """ Add the sensory input. """
        self.aff_inp = param

    def add_ode_input(self, int idx, neuron, neural_container, **kwargs):
        """ Add relevant external inputs to the output.
        Parameters
        ----------
        """

        # Create a struct to store the inputs and weights to the neuron
        cdef ReLUNeuronInput n
        # Get the neuron parameter
        neuron_idx = neural_container.outputs.get_parameter_index(
            'nout_'+neuron.n_id)

        # Add the weight parameter
        weight = neural_container.weights.add_parameter(
            'w_' + neuron.n_id + '_to_' + self.n_id, kwargs.get('weight', 0.0))[0]
        weight_idx = neural_container.weights.get_parameter_index(
            'w_' + neuron.n_id + '_to_' + self.n_id)
        n.neuron_idx = neuron_idx
        n.weight_idx = weight_idx

        # Append the struct to the list
        self.neuron_inputs[idx] = n

    def ode_rhs(self, y, w, p):
        """Abstract method"""
        self.c_ode_rhs(y, w, p)

    def output(self):
        """ Output of the neuron model.
        Returns
        ----------
        out: <cas.SX.sym>
            Output of the neuron  model
        """
        return self.c_output()

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
            _sum += (_neuron_out*_weight)
        self.ext_inp.c_set_value(_sum)

    cdef void c_output(self):
        """ Neuron output. """
        # Set the neuron output
        cdef double gain = self.gain.c_get_value()
        cdef double sign = self.sign.c_get_value()
        cdef double offset = self.offset.c_get_value()
        cdef double ext_in = self.ext_inp.c_get_value()
        cdef double res = gain*(sign*ext_in + offset)
        self.nout.c_set_value(max(0.0, res))
