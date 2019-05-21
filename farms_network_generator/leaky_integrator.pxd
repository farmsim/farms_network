"""Leaky Integrator Neuron."""

from farms_dae_generator.parameters cimport Param
cimport numpy as cnp
cimport cython

ctypedef double real

cdef class NeuronInput(object):
    cdef:
        LeakyIntegrator neuron
        Param weight

cdef class LeakyIntegrator(object):
    cdef:
        # dict __dict__
        readonly str n_id
        readonly str neuron_type

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
        real c_output(self)
        real c_neuron_input_eval(self, NeuronInput n)
