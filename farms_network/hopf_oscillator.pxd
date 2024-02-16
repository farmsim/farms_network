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

Hopf oscillator model

[1]L. Righetti and A. J. Ijspeert, “Pattern generators with sensory
feedback for the control of quadruped locomotion,” in 2008 IEEE
International Conference on Robotics and Automation, May 2008,
pp. 819–824. doi: 10.1109/ROBOT.2008.4543306.
"""

from farms_container.parameter cimport Parameter
from farms_network.neuron cimport Neuron

cdef struct HopfOscillatorNeuronInput:
    int neuron_idx
    int weight_idx

cdef class HopfOscillator(Neuron):
    cdef:
        readonly str n_id

        unsigned int num_inputs

        # parameters
        # constants
        double mu
        double omega
        double alpha
        double beta

        # states
        Parameter x
        Parameter y

        # inputs
        Parameter ext_in

        # ode
        Parameter xdot
        Parameter ydot

        # Ouputs
        Parameter nout

        # neuron connenctions
        HopfOscillatorNeuronInput[:] neuron_inputs

    cdef:
        void c_ode_rhs(self, double[:] _y, double[:] _w, double[:] _p)
        void c_output(self)
        double c_neuron_inputs_eval(self, double _neuron_out, double _weight)
