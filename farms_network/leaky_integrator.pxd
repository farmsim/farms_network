"""Leaky Integrator Neuron."""

from farms_container.parameter cimport Parameter
from farms_network.neuron cimport Neuron

cdef struct LeakyIntegratorNeuronInput:
    int neuron_idx
    int weight_idx

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
        Parameter m

        #: inputs
        Parameter ext_in

        #: ode
        Parameter mdot

        #: Ouputs
        Parameter nout

        #: neuron connenctions
        LeakyIntegratorNeuronInput[:] neuron_inputs

    cdef:
        void c_ode_rhs(self, double[:] _y, double[:] _w, double[:] _p) nogil
        void c_output(self) nogil
        double c_neuron_inputs_eval(self, double _neuron_out, double _weight) nogil
