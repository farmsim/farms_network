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

Rectified Linear Unit (ReLU) neurons.
"""

from farms_container.parameter cimport Parameter
from farms_network.neuron cimport Neuron


cdef struct ReLUNeuronInput:
    int neuron_idx
    int weight_idx


cdef class ReLUNeuron(Neuron):
    cdef:
        readonly str n_id

        unsigned int num_inputs

        # Parameters
        Parameter gain
        Parameter sign
        Parameter offset

        # Input from external system
        Parameter ext_inp

        # neuron connenctions
        ReLUNeuronInput[:] neuron_inputs

        # Ouputs
        Parameter nout

    cdef:
        void c_ode_rhs(self, double[:] _y, double[:] _w, double[:] _p)
        void c_output(self)
