"""Leaky Integrator Neuron."""

from farms_dae_generator.parameters cimport Param
from farms_network_generator.neuron cimport Neuron

cdef struct NeuronInput:
    cdef:
        int neuron_idx
        int weight_idx

cdef class LeakyIntegrator(Neuron):
    cdef:
        readonly str n_id

        unsigned int num_inputs

        #: parameters
        #: constants
        Param tau
        Param bias
        Param D

        #: states
        Param m

        #: inputs
        Param ext_in

        #: ode
        Param mdot

        #: neuron connenctions
        NeuronInput[:] neuron_inputs

    cdef:
        void c_ode_rhs(self)
        void c_output(self) nogil
        real c_neuron_input_eval(self, NeuronInput n) nogil
