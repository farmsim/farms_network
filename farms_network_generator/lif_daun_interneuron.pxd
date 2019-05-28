"""Leaky Integrate and Fire InterNeuron Based on Daun et.al."""

from farms_dae_generator.parameters cimport Param
from farms_network_generator.neuron cimport Neuron

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
        Param v
        Param h

        #: inputs
        Param g_app
        Param e_app

        #: ode
        Param vdot
        Param hdot

        #: Ouputs
        Param nout

        #: neuron connenctions
        DaunInterNeuronInput[:] neuron_inputs

    cdef:
        void c_ode_rhs(self, double[:] _y, double[:] _p)
        void c_output(self)
        double c_neuron_inputs_eval(self, double _neuron_out, double _g_syn, double _e_syn,
                                    double _gamma_s, double _v_h_s)
