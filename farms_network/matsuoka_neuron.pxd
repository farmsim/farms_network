"""Matsuoka Neuron model."""

from farms_container.parameter cimport Parameter
from farms_network.neuron cimport Neuron

cdef struct MatsuokaNeuronInput:
    int neuron_idx
    int weight_idx
    int phi_idx

cdef class MatsuokaNeuron(Neuron):
    cdef:
        readonly str n_id

        unsigned int num_inputs

        #: parameters
        #: constants
        double c
        double b
        double tau
        double T
        double theta

        #: states
        Parameter V
        Parameter w

        #: inputs
        Parameter ext_in

        #: ode
        Parameter V_dot
        Parameter w_dot

        #: Ouputs
        Parameter nout

        #: neuron connenctions
        MatsuokaNeuronInput[:] neuron_inputs

    cdef:
        void c_ode_rhs(self, double[:] _y, double[:] _p) nogil
        void c_output(self) nogil
        cdef double c_neuron_inputs_eval(
            self, double _neuron_out, double _weight, double _phi,
            double _V, double _w) nogil
