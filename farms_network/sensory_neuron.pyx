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

from farms_network.neuron import Neuron
from libc.stdio cimport printf

cdef class SensoryNeuron(Neuron):
    """Sensory afferent neurons connecting muscle model with the network.
    """

    def __init__(self, n_id, num_inputs, neural_container, **kwargs):
        """Initialize.
        Parameters
        ----------
        n_id: str
            Unique ID for the neuron in the network.
        """
        super(SensoryNeuron, self).__init__('sensory')

        # Neuron ID
        self.n_id = n_id

        self.aff_inp = neural_container.inputs.add_parameter(
            'aff_' + self.n_id, kwargs.get('init', 0.0))[0]

        # Output
        self.nout = neural_container.outputs.add_parameter(
            'nout_' + self.n_id, 0.0)[0]

    def reset_sensory_param(self, param):
        """ Add the sensory input. """
        self.aff_inp = param

    def add_ode_input(self, int idx, neuron, neural_container, **kwargs):
        """Abstract method"""
        pass

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
        pass

    cdef void c_output(self):
        """ Neuron output. """
        # Set the neuron output
        self.nout.c_set_value(self.aff_inp.c_get_value())
