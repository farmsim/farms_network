"""Oscillator model."""

from farms_dae.parameters cimport Param
from farms_network.neuron cimport Neuron

cdef struct OscillatorNeuronInput:
    int neuron_idx
    int weight_idx
    int phi_idx

cdef class Oscillator(Neuron):
    cdef:
        readonly str n_id

        unsigned int num_inputs

        #: parameters
        #: constants
        double f
        double R
        double a

        #: states
        Param phase
        Param amp

        #: inputs
        Param ext_in

        #: ode
        Param phase_dot
        Param amp_dot

        #: Ouputs
        Param nout

        #: neuron connenctions
        OscillatorNeuronInput[:] neuron_inputs

    cdef:
        void c_ode_rhs(self, double[:] _y, double[:] _p) nogil
        void c_output(self) nogil
        cdef double c_neuron_inputs_eval(
            self, double _neuron_out, double _weight, double _phi,
            double _phase, double _amp) nogil
