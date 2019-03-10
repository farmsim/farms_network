"""Leaky Integrate and Fire Neuron Based on Danner et.al."""

from neuron import Neuron
import casadi as cas
import numpy as np
import farms_pylog as biolog


class LIFDanner(Neuron):
    """Leaky Integrate and Fire Neuron Based on Danner et.al.
    """

    def __init__(self, n_id, dae, **kwargs):
        super(LIFDanner, self).__init__(neuron_type='lif_danner')
        self.n_id = n_id  #: Unique neuron identifier
        self.dae = dae

        #: Constants
        self.c_m = self.dae.add_c('c_m_' + self.n_id,
                                  kwargs.get('c_m', 10.0),
                                  param_type='val')  # : pF

        self.g_leak = self.dae.add_c('g_leak_' + self.n_id,
                                     kwargs.get('g_leak', 2.8),
                                     param_type='val')  #: nS
        self.e_leak = self.dae.add_c('e_leak_' + self.n_id,
                                     kwargs.get('e_leak', -60.0),
                                     param_type='val')  #: mV

        self.tau_noise = self.dae.add_c('tau_noise_' + self.n_id,
                                        kwargs.get('tau_noise', 10.0),
                                        param_type='val')  #: ms

        self.v_max = self.dae.add_c('v_max_' + self.n_id,
                                    kwargs.get('v_max', 0.0),
                                    param_type='val')  #: mV
        self.v_thr = self.dae.add_c('v_thr_' + self.n_id,
                                    kwargs.get('v_thr', -50.0),
                                    param_type='val')  #: mV

        self.g_syn_e = self.dae.add_c('g_syn_e_' + self.n_id,
                                      kwargs.get('g_syn_e', 10.0),
                                      param_type='val')  #: nS
        self.g_syn_i = self.dae.add_c('g_syn_i_' + self.n_id,
                                      kwargs.get('g_syn_i', 10.0),
                                      param_type='val')  #: nS
        self.e_syn_e = self.dae.add_c('e_syn_e_' + self.n_id,
                                      kwargs.get('e_syn_e', -10.0),
                                      param_type='val')  #: mV
        self.e_syn_i = self.dae.add_c('e_syn_i_' + self.n_id,
                                      kwargs.get('e_syn_i', -75.0),
                                      param_type='val')  #: mV

        #: State Variables
        #: Membrane potential
        #: pylint: disable=invalid-name
        self.v = self.dae.add_x('V_' + self.n_id,
                                kwargs.get('v0'))
        # self.i_noise = cas.SX.sym('In_' + self.n_id)

        #: ODE
        self.vdot = self.dae.add_ode('vdot_' + self.n_id, 0.0)

        #: External Input (BrainStem Drive)
        self.alpha = self.dae.add_u('alpha_' + self.n_id, 0.2)
        self.m_e = self.dae.add_c('m_e_' + self.n_id,
                                  kwargs.pop('m_e', 0.0),
                                  param_type='val')  #: m_E,i
        self.m_i = self.dae.add_c('m_i_' + self.n_id,
                                  kwargs.pop('m_i', 0.0),
                                  param_type='val')  #: m_I,i
        self.b_e = self.dae.add_c('b_e_' + self.n_id,
                                  kwargs.pop('b_e', 0.0),
                                  param_type='val')  #: m_E,i
        self.b_i = self.dae.add_c('b_i_' + self.n_id,
                                  kwargs.pop('b_i', 0.0),
                                  param_type='val')  #: m_I,i

        self.d_e = self.m_e.param * self.alpha.sym + self.b_e.param
        self.d_i = self.m_i.param * self.alpha.sym + self.b_i.param

        #: ODE

        self.ode_rhs()

        return

    def add_ode_input(self, neuron, **kwargs):
        """ Add relevant external inputs to the ode.
        Parameters
        ----------
        neuron : <LIF_Danner>
            Neuron model from which the input is received.
        weight : <float>
            Strength of the synapse between the two neurons
        """

        weight = self.dae.add_p(
            'w_' + neuron.n_id + '_to_' + self.n_id, kwargs.get(
                'weight'),
            param_type='val')

        #: Weight squashing function
        def s_w(val):
            """ Weight squashing internal function."""
            return val * (val >= 0.0)

        #: pylint: disable=no-member
        if np.sign(weight.val) == 1:
            #: Excitatory Synapse
            biolog.debug('Adding excitatory signal of weight {}'.format(
                s_w(weight.val)))
            self.vdot.sym += -(self.g_syn_e.param*(
                s_w(weight.param) * neuron.neuron_out())*(
                    self.v.sym - self.e_syn_e.param))/self.c_m.param
        elif np.sign(weight.val) == -1:
            #: Inhibitory Synapse
            biolog.debug('Adding inhibitory signal of weight {}'.format(
                s_w(-weight.val)))
            self.vdot.sym += -(self.g_syn_i.param*(
                s_w(-weight.param)*neuron.neuron_out())*(
                    self.v.sym - self.e_syn_i.param))/self.c_m.param
        return

    def ode_rhs(self):
        """Generate initial ode rhs."""

        #: Ileak
        i_leak = self.g_leak.param * (self.v.sym - self.e_leak.param)

        #: ISyn_Excitatory
        i_syn_e = self.g_syn_e.param * (self.d_e) * (
            self.v.sym - self.e_syn_e.param)

        #: ISyn_Inhibitory
        i_syn_i = self.g_syn_i.param * (self.d_i) * (
            self.v.sym - self.e_syn_i.param)

        #: dV
        self.vdot.sym = - (i_leak + i_syn_e + i_syn_i)/self.c_m.param
        return

    def neuron_out(self, res=None):
        """ Output of the neuron model."""
        if res is None:
            _cond = cas.logic_and(self.v_thr.param <= self.v.sym,
                                  self.v.sym < self.v_max.param)
            _f = (self.v.sym - self.v_thr.param) / \
                (self.v_max.param - self.v_thr.param)
            return cas.if_else(_cond, _f, 1.) * (
                self.v.sym > self.v_thr.param)
        else:
            _cond = cas.logic_and(self.v_thr.param <= res,
                                  res < self.v_max.param)
            _f = (res - self.v_thr.param) / \
                (self.v_max.param - self.v_thr.param)
            return cas.if_else(_cond, _f, 1.) * (
                res > self.v_thr.param)
