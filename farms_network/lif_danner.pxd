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
from libcpp.random cimport mt19937, normal_distribution

from farms_network.neuron cimport Neuron
from farms_network.utils.ornstein_uhlenbeck cimport OrnsteinUhlenbeckParameters

cdef struct DannerNeuronInput:
    int neuron_idx
    int weight_idx

cdef class LIFDanner(Neuron):
    cdef:
        readonly str n_id

        unsigned int num_inputs

        # parameters
        # constants
        double c_m
        double g_leak
        double e_leak
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

        double tau_noise
        double mu_noise
        double sigma_noise
        double time_step_noise
        unsigned long int seed_noise

        # states
        Parameter v

        Parameter state_noise

        # inputs
        Parameter alpha

        # ode
        Parameter vdot

        # Ouputs
        Parameter nout

        # neuron connenctions
        DannerNeuronInput[:] neuron_inputs

        # current noise
        OrnsteinUhlenbeckParameters noise_params
        mt19937 random_mt19937
        normal_distribution[double] distribution

    cdef:
        void c_ode_rhs(self, double[:] _y, double[:] _w, double[:] _p) nogil
        void c_output(self) nogil
        inline double c_neuron_inputs_eval(self, double _neuron_out, double _weight) nogil
