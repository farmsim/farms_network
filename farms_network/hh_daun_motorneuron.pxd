"""
----------------------------------------------------------------------
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

Hodgkin Huxley Motor Neuron Based on Daun et.al.
"""

from farms_container.parameter cimport Parameter
from farms_network.neuron cimport Neuron

cdef struct DaunMotorNeuronInput:
    int neuron_idx
    int g_syn_idx
    int e_syn_idx
    int gamma_s_idx
    int v_h_s_idx

cdef class HHDaunMotorneuron(Neuron):
    cdef:
        readonly str n_id

        unsigned int num_inputs

        #: parameters
        #: constants
        double g_nap
        double e_nap
        double am1_nap
        double am2_nap
        double am3_nap
        double bm1_nap
        double bm2_nap
        double bm3_nap
        double ah1_nap
        double ah2_nap
        double ah3_nap
        double bh1_nap
        double bh2_nap
        double bh3_nap

        #: Parameters of IK
        double g_k
        double e_k
        double am1_k
        double am2_k
        double am3_k
        double bm1_k
        double bm2_k
        double bm3_k

        #: Parameters of Iq
        double g_q
        double e_q
        double gamma_q
        double r_q
        double v_m_q

        #: Parameters of Ileak
        double g_leak
        double e_leak

        #: Parameters of Isyn
        double g_syn
        double e_syn
        double v_hs
        double gamma_s

        #: Other constants
        double c_m

        #: State Variables
        Parameter v
        Parameter m_na
        Parameter h_na
        Parameter m_k
        Parameter m_q

        #: ODE
        Parameter vdot
        Parameter m_na_dot
        Parameter h_na_dot
        Parameter m_k_dot
        Parameter m_q_dot

        #: External Input
        Parameter g_app
        Parameter e_app

        #: Output
        Parameter nout

        #: neuron connenctions
        DaunMotorNeuronInput[:] neuron_inputs

    cdef:
        void c_ode_rhs(self, double[:] _y, double[:] _w, double[:] _p) nogil
        void c_output(self) nogil
        double c_neuron_inputs_eval(self, double _neuron_out, double _g_syn, double _e_syn,
                                    double _gamma_s, double _v_h_s) nogil
