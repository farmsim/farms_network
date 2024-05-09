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

Leaky Integrator Neuron based on Danner et.al.
"""

from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strdup


cdef void ode_rhs_c(
    double time,
    double[:] states,
    double[:] dstates,
    double[:] inputs,
    double[:] weights,
    double[:] noise,
    double drive,
    Neuron neuron
):
    """ ODE """

    # # Parameters
    cdef LIDannerNeuronParameters params = (
        <LIDannerNeuronParameters*> neuron.parameters
    )[0]
    # States
    cdef double state_v = states[0]

    # External Modulation
    cdef double alpha = drive

    # Drive inputs
    cdef double d_e = params.m_e * alpha + params.b_e # Excitatory drive
    cdef double d_i = params.m_i * alpha + params.b_i # Inhibitory drive

    # Ileak
    cdef double i_leak = params.g_leak * (state_v - params.e_leak)

    # ISyn_Excitatory
    cdef double i_syn_e = params.g_syn_e * d_e * (state_v - params.e_syn_e)

    # ISyn_Inhibitory
    cdef double i_syn_i = params.g_syn_i * d_i * (state_v - params.e_syn_i)

    # Neuron inputs
    cdef double _sum = 0.0
    cdef unsigned int j
    cdef double _neuron_out
    cdef double _weight

    # for j in range(neuron.ninputs):
    #     _sum += neuron_inputs_eval_c(inputs[j], weights[j])

    # # noise current
    # cdef double i_noise = c_noise_current_update(
    #     self.state_noise.c_get_value(), &(self.noise_params)
    # )
    # self.state_noise.c_set_value(i_noise)

    # dV
    cdef i_noise = 0.0
    dstates[0] = (
        -(i_leak + i_syn_e + i_syn_i + i_noise + _sum)/params.c_m
    )


cdef double output_c(double time, double[:] states, Neuron neuron):
    """ Neuron output. """

    cdef double state_v = states[0]
    cdef double _n_out = 1000.0

    # cdef LIDannerNeuronParameters params = <LIDannerNeuronParameters> neuron.parameters

    # if state_v >= params.v_max:
    #     _n_out = 1.0
    # elif (params.v_thr <= state_v) and (state_v < params.v_max):
    #     _n_out = (state_v - params.v_thr) / (params.v_max - params.v_thr)
    # elif state_v < params.v_thr:
    #     _n_out = 0.0
    return _n_out


cdef double neuron_inputs_eval_c(double _neuron_out, double _weight):
    return 0.0


cdef class PyLIDannerNeuron(PyNeuron):
    """ Python interface to Leaky Integrator Neuron C-Structure """

    def __cinit__(self):
        self._neuron.model_type = strdup("LI_DANNER".encode('UTF-8'))
        # override default ode and out methods
        self._neuron.ode_rhs_c = ode_rhs_c
        self._neuron.output_c = output_c
        # parameters
        self._neuron.parameters = <LIDannerNeuronParameters*>malloc(
            sizeof(LIDannerNeuronParameters)
        )

    def __dealloc__(self):
        if self._neuron.name is not NULL:
            free(self._neuron.parameters)

    @classmethod
    def from_options(cls, neuron_options: NeuronOptions):
        """ From neuron options """
        name: str = neuron_options.name
        ninputs: int = neuron_options.ninputs
        return cls(name, ninputs)
