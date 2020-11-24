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

Morris Lecar Neuron model.
"""

from farms_container.parameter cimport Parameter
from farms_network.neuron cimport Neuron

cdef struct MLNeuronInput:
    int neuron_idx
    int weight_idx
    int phi_idx

cdef class MorrisLecarNeuron(Neuron):
    cdef:
        readonly str n_id

        unsigned int num_inputs

        #: parameters
        #: constants
        double I
        double C
        double g_fast
        double g_slow
        double g_leak
        double E_fast
        double E_slow
        double E_leak
        double phi_w
        double beta_m
        double beta_w
        double gamma_m
        double gamma_w

        #: states
        Parameter V
        Parameter w

        #: inputs
        Parameter ext_in

        #: ode
        Parameter V_dot
        Parameter w_dot

        #: Ouputs
        Parameter nout

        #: neuron connenctions
        MLNeuronInput[:] neuron_inputs

    cdef:
        void c_ode_rhs(self, double[:] _y, double[:] _w, double[:] _p) nogil
        void c_output(self) nogil
        cdef double c_neuron_inputs_eval(
            self, double _neuron_out, double _weight, double _phi,
            double _V, double _w) nogil
