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

Leaky Integrator Node Based on Danner et.al. with Na and K channels
"""

from ..core.node cimport Node, PyNode
from ..core.edge cimport Edge


cdef enum:

    #STATES
    NSTATES = 2
    STATE_V = 0
    STATE_H = 1


cdef packed struct LINaPDannerNodeParameters:

    double c_m                  # pF
    double g_leak               # nS
    double e_leak               # mV
    double g_nap                # nS
    double e_na                 # mV
    double g_syn_e              # nS
    double g_syn_i              # nS
    double e_syn_e              # mV
    double e_syn_i              # mV
    double v1_2_m               # mV
    double k_m                  #
    double v1_2_h               # mV
    double k_h                  #
    double v1_2_t               # mV
    double k_t                  #
    double tau_0                # mS
    double tau_max              # mS
    double v_max                # mV
    double v_thr                # mV


cdef:
    void ode(
        double time,
        double* states,
        double* derivatives,
        double external_input,
        double* network_outputs,
        unsigned int* inputs,
        double* weights,
        Node* node,
        Edge* edges,
    ) noexcept
    double output(
        double time,
        double* states,
        double external_input,
        double* network_outputs,
        unsigned int* inputs,
        double* weights,
        Node* node,
        Edge* edges,
    ) noexcept


cdef class PyLINaPDannerNode(PyNode):
    """ Python interface to Leaky Integrator Node C-Structure """

    cdef:
        LINaPDannerNodeParameters parameters
