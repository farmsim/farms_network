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

Leaky Integrator Node Based on Danner et.al.
"""

from ..core.node cimport Node, PyNode
from ..core.edge cimport Edge


cdef enum:

    #STATES
    NSTATES = 1
    STATE_V = 0


cdef packed struct LIDannerNodeParameters:

    double c_m                     # pF
    double g_leak                  # nS
    double e_leak                  # mV
    double v_max                   # mV
    double v_thr                   # mV
    double g_syn_e                 # nS
    double g_syn_i                 # nS
    double e_syn_e                 # mV
    double e_syn_i                 # mV


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


cdef class PyLIDannerNode(PyNode):
    """ Python interface to Leaky Integrator Node C-Structure """

    cdef:
        LIDannerNodeParameters parameters
