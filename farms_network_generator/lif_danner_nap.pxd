"""Leaky Integrate and Fire Neuron Based on Danner et.al."""

from farms_dae_generator.parameters cimport Param
from farms_network_generator.neuron cimport Neuron

cdef struct DannerNapNeuronInput:
    int neuron_idx
    int weight_idx

cdef class LIFDannerNap(Neuron):
    cdef:
        readonly str n_id

        unsigned int num_inputs

        #: parameters
        #: constants
        double c_m
        double g_nap
        double e_na
        double v1_2_m
        double k_m
        double v1_2_h
        double k_h
        double v1_2_t
        double k_t
        double g_leak
        double e_leak
        double tau_0
        double tau_max
        double tau_noise
        double v_max
        double v_thr
        double g_syn_e
        double g_syn_i
        double e_syn_e
        double e_syn_i
        double m_e
        double m_i
        double b_e
        double b_i

        #: states
        Param v
        Param h

        #: inputs
        Param alpha

        #: ode
        Param vdot
        Param hdot

        #: Ouputs
        Param nout

        #: neuron connenctions
        DannerNapNeuronInput[:] neuron_inputs

    cdef:
        void c_ode_rhs(self, double[:] _y, double[:] _p) nogil
        void c_output(self) nogil
        double c_neuron_inputs_eval(self, double _neuron_out, double _weight) nogil
