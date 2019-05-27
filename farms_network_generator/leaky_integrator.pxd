"""Leaky Integrator Neuron."""

from farms_dae_generator.parameters cimport Param
from farms_network_generator.neuron cimport Neuron

cdef struct LeakyIntegratorNeuronInput:
    unsigned int neuron_idx
    unsigned int weight_idx

cdef class LeakyIntegrator(Neuron):
    cdef:
        readonly str n_id

        unsigned int num_inputs

        #: parameters
        #: constants
        double tau
        double bias
        double D

        #: states
        Param m

        #: inputs
        Param ext_in

        #: ode
        Param mdot

        #: Ouputs
        Param nout

        #: neuron connenctions
        LeakyIntegratorNeuronInput[:] neuron_inputs

    cdef:
        void c_ode_rhs(self, double[:] _y, double[:] _p)
        void c_output(self)
        double c_neuron_inputs_eval(self, double _neuron_out, double _weight)
