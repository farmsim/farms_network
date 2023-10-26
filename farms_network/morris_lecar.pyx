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

Morris Lecar Neuron model
"""
from libc.stdio cimport printf
import farms_pylog as pylog
from libc.math cimport tanh as ctanh
from libc.math cimport cosh as ccosh
import numpy as np
cimport numpy as cnp


cdef class MorrisLecarNeuron(Neuron):

    def __init__(self, n_id, num_inputs, neural_container, **kwargs):
        """Initialize.

        Parameters
        ----------
        n_id: str
            Unique ID for the neuron in the network.
        """
        super(MorrisLecarNeuron, self).__init__('morris_lecar')

        #: Neuron ID
        self.n_id = n_id

        #: Initialize parameters
        (_, self.internal_curr) = neural_container.constants.add_parameter(
            'I_' + self.n_id, kwargs.get('I', 100.0))
        (_, self.C) = neural_container.constants.add_parameter(
            'C_' + self.n_id, kwargs.get('C', 2.0))
        (_, self.g_fast) = neural_container.constants.add_parameter(
            'g_fast_' + self.n_id, kwargs.get('g_fast', 20.0))
        (_, self.g_slow) = neural_container.constants.add_parameter(
            'g_slow_' + self.n_id, kwargs.get('g_slow', 20.0))
        (_, self.g_leak) = neural_container.constants.add_parameter(
            'g_leak_' + self.n_id, kwargs.get('g_leak', 2.0))
        (_, self.E_fast) = neural_container.constants.add_parameter(
            'E_fast_' + self.n_id, kwargs.get('E_fast', 50.0))
        (_, self.E_slow) = neural_container.constants.add_parameter(
            'E_slow_' + self.n_id, kwargs.get('E_slow', -100.0))
        (_, self.E_leak) = neural_container.constants.add_parameter(
            'E_leak_' + self.n_id, kwargs.get('E_leak', -70.0))
        (_, self.phi_w) = neural_container.constants.add_parameter(
            'phi_w_' + self.n_id, kwargs.get('phi_w', 0.15))
        (_, self.beta_m) = neural_container.constants.add_parameter(
            'beta_m_' + self.n_id, kwargs.get('beta_m', 0.0))
        (_, self.gamma_m) = neural_container.constants.add_parameter(
            'gamma_m_' + self.n_id, kwargs.get('gamma_m', 18.0))
        (_, self.beta_w) = neural_container.constants.add_parameter(
            'beta_w_' + self.n_id, kwargs.get('beta_w', -10.0))
        (_, self.gamma_w) = neural_container.constants.add_parameter(
            'gamma_w_' + self.n_id, kwargs.get('gamma_w', 13.0))

        #: Initialize states
        self.V = neural_container.states.add_parameter(
            'V_' + self.n_id, kwargs.get('V0', 0.0))[0]
        self.w = neural_container.states.add_parameter(
            'w_' + self.n_id, kwargs.get('w0', 0.0))[0]

        #: External inputs
        self.ext_in = neural_container.inputs.add_parameter(
            'ext_in_' + self.n_id)[0]

        #: ODE RHS
        self.V_dot = neural_container.dstates.add_parameter(
            'V_dot_' + self.n_id, 0.0)[0]
        self.w_dot = neural_container.dstates.add_parameter(
            'w_dot_' + self.n_id, 0.0)[0]

        #: Output
        self.nout = neural_container.outputs.add_parameter(
            'nout_' + self.n_id, 0.0)[0]

        #: Neuron inputs
        self.neuron_inputs = cnp.ndarray((num_inputs,),
                                         dtype=[('neuron_idx', 'i'),
                                                ('weight_idx', 'i'),
                                                ('phi_idx', 'i')])

        self.num_inputs = num_inputs

    def add_ode_input(self, int idx, neuron, neural_container, **kwargs):
        """ Add relevant external inputs to the ode."""
        #: Create a struct to store the inputs and weights to the neuron
        cdef MLNeuronInput n
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
        cdef double _V = self.V.c_get_value()
        cdef double _W = self.w.c_get_value()

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
            _sum += self.c_neuron_inputs_eval(_neuron_out,
                                              _weight, _phi, _V, _W)

        cdef double m_inf_V = 0.5*(1.0+ctanh((_V-self.beta_m)/self.gamma_m))

        cdef double w_inf_V = 0.5*(1.0+ctanh((_V-self.beta_w)/self.gamma_w))

        cdef double tau_w_V = (1./ccosh((_V-self.beta_w)/(2*self.gamma_w)))

        # V_dot
        self.V_dot.c_set_value((1.0/self.C)*(self.internal_curr - self.g_fast*m_inf_V*(_V-self.E_fast)
                                             - self.g_slow*_W*(_V - self.E_slow) - self.g_leak*(_V - self.E_leak)))

        #: wdot
        self.w_dot.c_set_value(self.phi_w*(w_inf_V - _W)/tau_w_V)

    cdef void c_output(self) nogil:
        """ Neuron output. """
        self.nout.c_set_value(self.V.c_get_value())

    cdef double c_neuron_inputs_eval(
            self, double _neuron_out, double _weight, double _phi,
            double _V, double _w) nogil:
        """ Evaluate neuron inputs."""
        # Linear coupling term in potential
        return _weight*(_neuron_out - _V)
