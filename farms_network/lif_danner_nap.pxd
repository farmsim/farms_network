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

Leaky Integrate and Fire Neuron Based on Danner et.al.
"""

from farms_container.parameter cimport Parameter
from farms_network.neuron cimport Neuron

cdef struct DannerNapNeuronInput:
    int neuron_idx
    int weight_idx

cdef class LIFDannerNap(Neuron):
    cdef:
        readonly str n_id

        unsigned int num_inputs

        #: parameters
        #: constants
        double c_m
        double g_nap
        double e_na
        double v1_2_m
        double k_m
        double v1_2_h
        double k_h
        double v1_2_t
        double k_t
        double g_leak
        double e_leak
        double tau_0
        double tau_max
        double tau_noise
        double v_max
        double v_thr
        double g_syn_e
        double g_syn_i
        double e_syn_e
        double e_syn_i
        double m_e
        double m_i
        double b_e
        double b_i

        #: states
        Parameter v
        Parameter h

        #: inputs
        Parameter alpha

        #: ode
        Parameter vdot
        Parameter hdot

        #: Ouputs
        Parameter nout

        #: neuron connenctions
        DannerNapNeuronInput[:] neuron_inputs

    cdef:
        void c_ode_rhs(self, double[:] _y, double[:] _w, double[:] _p) nogil
        void c_output(self) nogil
        double c_neuron_inputs_eval(self, double _neuron_out, double _weight) nogil
