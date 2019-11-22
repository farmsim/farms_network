"""Leaky Integrate and Fire InterNeuron Based on Daun et.al."""

from farms_container.parameter cimport Parameter
from farms_network.neuron cimport Neuron

cdef struct DaunInterNeuronInput:
    int neuron_idx
    int g_syn_idx
    int e_syn_idx
    int gamma_s_idx
    int v_h_s_idx

cdef class LIFDaunInterneuron(Neuron):
    cdef:
        readonly str n_id

        unsigned int num_inputs

        #: parameters
        #: constants
        double c_m
        double g_nap
        double e_nap
        double v_h_h
        double gamma_h
        double v_t_h
        double eps
        double gamma_t
        double v_h_m
        double gamma_m
        double g_leak
        double e_leak

        #: states
        Parameter v
        Parameter h

        #: inputs
        Parameter g_app
        Parameter e_app

        #: ode
        Parameter vdot
        Parameter hdot

        #: Ouputs
        Parameter nout

        #: neuron connenctions
        DaunInterNeuronInput[:] neuron_inputs

    cdef:
        void c_ode_rhs(self, double[:] _y, double[:] _w, double[:] _p) nogil
        void c_output(self) nogil
        double c_neuron_inputs_eval(self, double _neuron_out, double _g_syn, double _e_syn,
                                    double _gamma_s, double _v_h_s) nogil
