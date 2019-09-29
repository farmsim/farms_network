"""Sensory afferent neurons."""

from farms_dae.parameters cimport Param
from farms_network.neuron cimport Neuron

cdef class SensoryNeuron(Neuron):
    cdef:
        readonly str n_id
        
        #: Input from external system
        Param aff_inp

        #: Ouputs
        Param nout

    cdef:
        void c_ode_rhs(self, double[:] _y, double[:] _p) nogil
        void c_output(self) nogil
