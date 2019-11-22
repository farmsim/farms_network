"""Sensory afferent neurons."""

from farms_container.parameter cimport Parameter
from farms_network.neuron cimport Neuron

cdef class SensoryNeuron(Neuron):
    cdef:
        readonly str n_id
        
        #: Input from external system
        Parameter aff_inp

        #: Ouputs
        Parameter nout

    cdef:
        void c_ode_rhs(self, double[:] _y, double[:] _w, double[:] _p) nogil
        void c_output(self) nogil
