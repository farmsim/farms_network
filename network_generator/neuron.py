"""Implementation of different neuron models."""
import casadi as cas
import casadi.tools as cast
import numpy as np

import biolog


class Neuron(object):
    """Base neuron class.
    Inherits from Casadi Dae Builder Class.
    """

    def __init__(self, neuron_type):
        super(Neuron, self).__init__()
        self._neuron_type = neuron_type  # : Type of neuron

    @property
    def neuron_type(self):
        """Neuron type.  """
        return self._neuron_type

    @neuron_type.setter
    def neuron_type(self, value):
        """
        Parameters
        ----------
        value : <str>
            Type of neuron model
        """
        self._neuron_type = value

    def add_ode_input(self, neuron, **kwargs):
        """Add relevant external inputs to the ode.
        Parameters
        ----------
        neuron : <LIF_Danner>
            Neuron model from which the input is received.
        kwargs : <dict>
             Contains the weight/synaptic information from the receiving neuron.
        """
        raise NotImplementedError(
            'add_ode_input : Method not implemented in child class')

    def ode_rhs(self):
        """ ODE RHS.
        Returns
        ----------
        ode_rhs: <list>
            List containing the rhs equations of the ode states in the system
        """
        raise NotImplementedError(
            'ode_rhs : Method not implemented in child class')

    def ode_alg_eqn(self):
        """ ODE Algebraic equations.
        Returns
        ----------
        alg_eqn: <list>
            List containing the ode algebraic equations
        """
        raise NotImplementedError(
            'ode_alg_eqn : Method not implemented in child class')

    def neuron_out(self):
        """ Output of the neuron model.
        Returns
        ----------
        out: <cas.SX.sym>
            Output of the neuron  model
        """
        raise NotImplementedError(
            'neuron_out : Method not implemented in child class')


class IntegrateAndFire(Neuron):
    """Integrate & Fire Neuron Model."""

    def __init__(self, n_id, dae, **kwargs):
        """Initialize.

        Parameters
        ----------
        n_id: str
            Unique ID for the neuron in the network.
        """
        super(IntegrateAndFire, self).__init__(
            neuron_type='if')

        #: Neuron ID
        self.n_id = n_id
        self.dae = dae

        #: Initialize parameters
        self.tau = self.dae.add_c('tau_' + self.n_id,
                                  kwargs.get('tau', 0.1))
        self.bias = self.dae.add_c('bias_' + self.n_id,
                                   kwargs.get('bias', -2.75))
        #: pylint: disable=invalid-name
        self.D = self.dae.add_c('D_' + self.n_id,
                                kwargs.get('D', 1.0))

        #: Initialize states
        self.m = self.dae.add_x('m_' + self.n_id,
                                kwargs.get('x0', 0.0))
        #: External inputs
        self.ext_in = self.dae.add_u('ext_in_' + self.n_id)

        #: ODE RHS
        self.mdot = self.dae.add_ode('mdot_' + self.n_id, 0.0)
        self.ode_rhs()

    def add_ode_input(self, neuron, **kwargs):
        """ Add relevant external inputs to the ode."""
        weight = self.dae.add_p(
            'w_' + neuron.n_id + '_to_' + self.n_id,
            kwargs.get('weight'))
        self.mdot.sym += (
            neuron.neuron_out()*weight.sym)/self.tau.sym
        return

    def ode_rhs(self):
        """ Generate the ODE. Internal Setup Function."""
        self.mdot.sym = (
            -self.m.sym + self.ext_in.sym)/self.tau.sym

    def ode_alg_eqn(self):
        """ Abstract class. """
        pass

    def neuron_out(self):
        """Neuron activation function.
        Parameters
        ----------
        m_potential: float
            Neuron membrane potential
        """
        exp = np.exp  # pylint: disable = no-member
        return 1. / (1. + exp(-self.D.sym * (
            self.m.sym + self.bias.sym)))


class LIF_Danner_Nap(Neuron):
    """Leaky Integrate and Fire Neuron Based on Danner et.al.
    """

    def __init__(self, n_id, dae, **kwargs):
        super(
            LIF_Danner_Nap, self).__init__(neuron_type='lif_danner_nap')

        self.n_id = n_id  #: Unique neuron identifier
        self.dae = dae  #: Container for network

        #: Constants
        self.c_m = self.dae.add_c('c_m_' + self.n_id,
                                  kwargs.get('c_m', 10.0),
                                  param_type='val')  #: pF

        self.g_nap = self.dae.add_c(
            'g_nap_'+self.n_id, kwargs.get('g_nap', 4.5),
            param_type='val')  #: nS
        self.e_na = self.dae.add_c(
            'e_na_'+self.n_id, kwargs.get('e_na', 50.0),
            param_type='val')  #: mV

        self.v1_2_m = self.dae.add_c(
            'v1_2_m_' + self.n_id, kwargs.get('v1_2_m', -40.0),
            param_type='val')  #: mV
        self.k_m = self.dae.add_c(
            'k_m_' + self.n_id, kwargs.get('k_m', -6.0),
            param_type='val')  #: mV

        self.v1_2_h = self.dae.add_c(
            'v1_2_h_' + self.n_id, kwargs.get('v1_2_h', -45.0),
            param_type='val')  #: mV
        self.k_h = self.dae.add_c(
            'k_h_' + self.n_id, kwargs.get('k_h', 4.0),
            param_type='val')  #: mV

        self.v1_2_t = self.dae.add_c(
            'v1_2_t_' + self.n_id, kwargs.get('v1_2_t', -35.0),
            param_type='val')  #: mV
        self.k_t = self.dae.add_c(
            'k_t_' + self.n_id, kwargs.get('k_t', 15.0),
            param_type='val')  #: mV

        self.g_leak = self.dae.add_c(
            'g_leak_' + self.n_id, kwargs.get('g_leak', 4.5),
            param_type='val')  #: nS
        self.e_leak = self.dae.add_c(
            'e_leak_' + self.n_id, kwargs.get('e_leak', -62.5),
            param_type='val')  #: mV

        self.tau_0 = self.dae.add_c(
            'tau_0_' + self.n_id, kwargs.get('tau_0', 80.0),
            param_type='val')  #: ms
        self.tau_max = self.dae.add_c(
            'tau_max_' + self.n_id, kwargs.get('tau_max', 160.0),
            param_type='val')  #: ms
        self.tau_noise = self.dae.add_c(
            'tau_noise_' + self.n_id, kwargs.get('tau_noise', 10.0),
            param_type='val')  #: ms

        self.v_max = self.dae.add_c(
            'v_max_' + self.n_id, kwargs.get('v_max', 0.0),
            param_type='val')  #: mV
        self.v_thr = self.dae.add_c(
            'v_thr_' + self.n_id, kwargs.get('v_thr', -50.0),
            param_type='val')  #: mV

        self.g_syn_e = self.dae.add_c(
            'g_syn_e_' + self.n_id, kwargs.get('g_syn_e', 10.0),
            param_type='val')  #: nS
        self.g_syn_i = self.dae.add_c(
            'g_syn_i_' + self.n_id, kwargs.get('g_syn_i', 10.0),
            param_type='val')  #: nS
        self.e_syn_e = self.dae.add_c(
            'e_syn_e_' + self.n_id, kwargs.get('e_syn_e', -10.0),
            param_type='val')  #: mV
        self.e_syn_i = self.dae.add_c(
            'e_syn_i_' + self.n_id, kwargs.get('e_syn_i', -75.0),
            param_type='val')  #: mV

        #: State Variables
        #: pylint: disable=invalid-name
        self.v = self.dae.add_x('V_' + self.n_id,
                                kwargs.get('v0'))  #: Membrane potential
        self.h = self.dae.add_x('h_' + self.n_id,
                                kwargs.get('h0'))
        # self.i_noise = self.dae.add_x('In_' + self.n_id)

        #: ODE
        self.vdot = self.dae.add_ode('vdot_' + self.n_id, 0.0)
        self.hdot = self.dae.add_ode('hdot_' + self.n_id, 0.0)

        #: External Input (BrainStem Drive)
        self.alpha = self.dae.add_u('alpha_' + self.n_id, 0.22)
        self.m_e = self.dae.add_c(
            'm_e_' + self.n_id, kwargs.pop('m_e', 0.0),
            param_type='val')  #: m_E,i
        self.m_i = self.dae.add_c(
            'm_i_' + self.n_id, kwargs.pop('m_i', 0.0),
            param_type='val')  #: m_I,i
        self.b_e = self.dae.add_c(
            'b_e_' + self.n_id, kwargs.pop('b_e', 0.0),
            param_type='val')  #: m_E,i
        self.b_i = self.dae.add_c(
            'b_i_' + self.n_id, kwargs.pop('b_i', 0.0),
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
            param_type='sym')

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

        #: tau_h(V)
        tau_h = self.tau_0.param + (self.tau_max.param - self.tau_0.param) / \
            cas.cosh((self.v.sym - self.v1_2_t.param) / self.k_t.param)

        #: h_inf(V)
        h_inf = cas.inv(
            1.0 + cas.exp((self.v.sym - self.v1_2_h.param) / self.k_h.param))

        #: Slow inactivation
        self.hdot.sym = (h_inf - self.h.sym) / tau_h

        #: m(V)
        m = cas.inv(
            1.0 + cas.exp((self.v.sym - self.v1_2_m.param) / self.k_m.param))

        #: Inap
        #: pylint: disable=no-member
        i_nap = self.g_nap.param * m * self.h.sym * \
            (self.v.sym - self.e_na.param)

        #: Ileak
        i_leak = self.g_leak.param * (self.v.sym - self.e_leak.param)

        #: ISyn_Excitatory
        i_syn_e = self.g_syn_e.param * self.d_e * \
            (self.v.sym - self.e_syn_e.param)

        #: ISyn_Inhibitory
        i_syn_i = self.g_syn_i.param * self.d_i * \
            (self.v.sym - self.e_syn_i.param)

        #: dV
        self.vdot.sym = -(i_nap + i_leak + i_syn_e + i_syn_i)/self.c_m.param
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


class LIF_Danner(Neuron):
    """Leaky Integrate and Fire Neuron Based on Danner et.al.
    """

    def __init__(self, n_id, dae, **kwargs):
        super(LIF_Danner, self).__init__(neuron_type='lif_danner')
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


class SensoryDanner(Neuron):
    """Sensory afferent neurons connecting muscle model with the network.
    """

    def __init__(self, n_id, dae, **kwargs):
        """Initialize.
        Parameters
        ----------
        n_id: str
            Unique ID for the neuron in the network.
        """
        super(SensoryDanner, self).__init__(
            neuron_type='sensory_danner')

        #: Neuron ID
        self.n_id = n_id
        self.dae = dae

        #: Initialize parameters
        self.weight = self.dae.add_p('weight_' + self.n_id,
                                     kwargs.get('weight', 0.1))

        return

    def add_ode_input(self, neuron, **kwargs):
        """Add relevant external inputs to the ode.
        Parameters
        ----------
        neuron : <LIF_Danner>
            Neuron model from which the input is received.
        kwargs : <dict>
             Contains the weight/synaptic information from the receiving neuron.
        """
        raise NotImplementedError(
            'add_ode_input : Method not implemented in child class')

    def ode_rhs(self):
        """ ODE RHS.
        Returns
        ----------
        ode_rhs: <list>
            List containing the rhs equations of the ode states in the system
        """
        raise NotImplementedError(
            'ode_rhs : Method not implemented in child class')

    def ode_alg_eqn(self):
        """ ODE Algebraic equations.
        Returns
        ----------
        alg_eqn: <list>
            List containing the ode algebraic equations
        """
        raise NotImplementedError(
            'ode_alg_eqn : Method not implemented in child class')

    def neuron_out(self):
        """ Output of the neuron model.
        Returns
        ----------
        out: <cas.SX.sym>
            Output of the neuron  model
        """
        raise NotImplementedError(
            'neuron_out : Method not implemented in child class')


class LIF_Daun_Interneuron(Neuron):
    """Leaky Integrate and Fire Interneuron.
    Based on Silvia Daun and Tbor's model.
    """

    def __init__(self, n_id, dae, **kwargs):
        super(LIF_Daun_Interneuron, self).__init__(
            neuron_type='lif_daun_interneuron')

        self.n_id = n_id  #: Unique neuron identifier
        self.dae = dae

        #: Constants
        self.g_nap = self.dae.add_c('g_nap_' + self.n_id,
                                    kwargs.get('g_nap', 10.0))
        self.e_nap = self.dae.add_c('e_nap_' + self.n_id,
                                    kwargs.get('e_nap', 50.0))

        #: Parameters of h
        self.v_h_h = self.dae.add_c('v_h_h_' + self.n_id,
                                    kwargs.get('v_h_h', -30.0))
        self.gamma_h = self.dae.add_c('gamma_h_' + self.n_id,
                                      kwargs.get('gamma_h', 0.1667))

        #: Parameters of tau
        self.v_t_h = self.dae.add_c('v_t_h_' + self.n_id,
                                    kwargs.get('v_t_h', -30.0))
        self.eps = self.dae.add_c('eps_' + self.n_id,
                                  kwargs.get('eps', 0.0023))
        self.gamma_t = self.dae.add_c('gamma_t_' + self.n_id,
                                      kwargs.get('gamma_t', 0.0833))

        #: Parameters of m
        self.v_h_m = self.dae.add_c('v_h_m_' + self.n_id,
                                    kwargs.get('v_h_m', -37.0))
        self.gamma_m = self.dae.add_c('gamma_m_' + self.n_id,
                                      kwargs.get('gamma_m', -0.1667))

        #: Parameters of Ileak
        self.g_leak = self.dae.add_c('g_leak_' + self.n_id,
                                     kwargs.get('g_leak', 2.8))
        self.e_leak = self.dae.add_c('e_leak_' + self.n_id,
                                     kwargs.get('e_leak', -65.0))

        #: Other constants
        self.c_m = self.dae.add_c('c_m_' + self.n_id,
                                  kwargs.get('c_m', 0.9154))

        #: State Variables
        #: pylint: disable=invalid-name
        #: Membrane potential
        self.v = self.dae.add_x('V_' + self.n_id,
                                kwargs.get('v0', -60.0))
        self.h = self.dae.add_x('h_' + self.n_id,
                                kwargs.get('h0', 0.0))

        #: ODE
        self.vdot = self.dae.add_ode('vdot_' + self.n_id, 0.0)
        self.hdot = self.dae.add_ode('hdot_' + self.n_id, 0.0)

        #: External Input
        self.g_app = self.dae.add_u('g_app_' + self.n_id,
                                    kwargs.get('g_app', 0.2))
        self.e_app = self.dae.add_u('e_app_' + self.n_id,
                                    kwargs.get('e_app', 0.0))

        #: Add outputs
        if kwargs.get('output'):
            self.dae.add_y(self.v)
            self.dae.add_y(self.h)

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
            Strength of the synapse between the two neurons"""

        g_syn = self.dae.add_p('g_syn_' + self.n_id,
                               kwargs.pop('g_syn', 0.0))
        e_syn = self.dae.add_p('e_syn_' + self.n_id,
                               kwargs.pop('e_syn', 0.0))
        gamma_s = self.dae.add_p('gamma_s_' + self.n_id,
                                 kwargs.pop('gamma_s', 0.0))
        v_h_s = self.dae.add_p('v_h_s_' + self.n_id,
                               kwargs.pop('v_h_s', 0.0))

        s_inf = cas.inv(
            1. + cas.exp(gamma_s.sym*(
                neuron.neuron_out() - v_h_s.sym)))

        self.vdot.sym += -(
            g_syn.sym*s_inf*(self.v.sym - e_syn.sym))/self.c_m.sym
        return

    def ode_rhs(self):
        """Generate initial ode rhs."""

        #: tau_h(V)
        tau_h = cas.inv(
            self.eps.sym*cas.cosh(
                self.gamma_t.sym*(self.v.sym - self.v_t_h.sym)))

        #: h_inf(V)
        h_inf = cas.inv(
            1. + cas.exp(
                self.gamma_h.sym*(self.v.sym - self.v_h_h.sym)))

        #: Slow inactivation
        self.hdot.sym = (h_inf - self.h.sym)/tau_h

        #: m_inf(V)
        m_inf = cas.inv(
            1. + cas.exp(
                self.gamma_m.sym*(self.v.sym - self.v_h_m.sym)))

        #: Inap
        #: pylint: disable=no-member
        i_nap = self.g_nap.sym * m_inf * self.h.sym * (
            self.v.sym - self.e_nap.sym)

        #: Ileak
        i_leak = self.g_leak.sym * (self.v.sym - self.e_leak.sym)

        #: Iapp
        i_app = self.g_app.sym * (self.v.sym - self.e_app.sym)

        #: dV
        self.vdot.sym = -(
            i_nap + i_leak + i_app)/self.c_m.sym
        return

    def ode_alg_eqn(self):
        """ ODE Algebraic Variables."""
        pass

    def neuron_out(self):
        """Neuron activation function.
        Parameters
        ----------
        m_potential: float
            Neuron membrane potential
        """
        return self.v.sym


class LIF_Daun_Motorneuron(Neuron):
    """Leaky Integrate and Fire Interneuron.
    Based on Silvia Daun and Tbor's model.
    """

    def __init__(self, n_id, dae, **kwargs):
        super(LIF_Daun_Motorneuron, self).__init__(
            neuron_type='lif_daun_motorneuron')

        self.n_id = n_id  #: Unique neuron identifier
        self.dae = dae

        #: Neuron constants
        #: Parameters of INaP
        self.g_nap = self.dae.add_c('g_nap' + self.n_id,
                                    kwargs.get('g_nap', 10.0))
        self.e_nap = self.dae.add_c('e_nap' + self.n_id,
                                    kwargs.get('e_nap', 55.0))
        self.am1_nap = self.dae.add_c('am1_nap' + self.n_id,
                                      kwargs.get('am1_nap', 0.32))
        self.am2_nap = self.dae.add_c('am2_nap' + self.n_id,
                                      kwargs.get('am2_nap', -51.90))
        self.am3_nap = self.dae.add_c('am3_nap' + self.n_id,
                                      kwargs.get('am3_nap', 0.25))
        self.bm1_nap = self.dae.add_c('bm1_nap' + self.n_id,
                                      kwargs.get('bm1_nap', -0.280))
        self.bm2_nap = self.dae.add_c('bm2_nap' + self.n_id,
                                      kwargs.get('bm2_nap', -24.90))
        self.bm3_nap = self.dae.add_c('bm3_nap' + self.n_id,
                                      kwargs.get('bm3_nap', -0.2))
        self.ah1_nap = self.dae.add_c('ah1_nap' + self.n_id,
                                      kwargs.get('ah1_nap', 0.1280))
        self.ah2_nap = self.dae.add_c('ah2_nap' + self.n_id,
                                      kwargs.get('ah2_nap', -48.0))
        self.ah3_nap = self.dae.add_c('ah3_nap' + self.n_id,
                                      kwargs.get('ah3_nap', 0.0556))
        self.bh1_nap = self.dae.add_c('bh1_nap' + self.n_id,
                                      kwargs.get('bh1_nap', 4.0))
        self.bh2_nap = self.dae.add_c('bh2_nap' + self.n_id,
                                      kwargs.get('bh2_nap', -25.0))
        self.bh3_nap = self.dae.add_c('bh3_nap' + self.n_id,
                                      kwargs.get('bh3_nap', 0.20))

        #: Parameters of IK
        self.g_k = self.dae.add_c('g_k' + self.n_id,
                                  kwargs.get('g_k', 2.0))
        self.e_k = self.dae.add_c('e_k' + self.n_id,
                                  kwargs.get('e_k', -80.0))
        self.am1_k = self.dae.add_c('am1_k' + self.n_id,
                                    kwargs.get('am1_k', 0.0160))
        self.am2_k = self.dae.add_c('am2_k' + self.n_id,
                                    kwargs.get('am2_k', -29.90))
        self.am3_k = self.dae.add_c('am3_k' + self.n_id,
                                    kwargs.get('am3_k', 0.20))
        self.bm1_k = self.dae.add_c('bm1_k' + self.n_id,
                                    kwargs.get('bm1_k', 0.250))
        self.bm2_k = self.dae.add_c('bm2_k' + self.n_id,
                                    kwargs.get('bm2_k', -45.0))
        self.bm3_k = self.dae.add_c('bm3_k' + self.n_id,
                                    kwargs.get('bm3_k', 0.025))

        #: Parameters of Iq
        self.g_q = self.dae.add_c('g_q' + self.n_id,
                                  kwargs.get('g_q', 12.0))
        self.e_q = self.dae.add_c('e_q' + self.n_id,
                                  kwargs.get('e_q', -80.0))
        self.gamma_q = self.dae.add_c('gamma_q' + self.n_id,
                                      kwargs.get('gamma_q', -0.6))
        self.r_q = self.dae.add_c('r_q' + self.n_id,
                                  kwargs.get('r_q', 0.0005))
        self.v_m_q = self.dae.add_c('v_m_q' + self.n_id,
                                    kwargs.get('v_m_q', -30.0))

        #: Parameters of Ileak
        self.g_leak = self.dae.add_c('g_leak' + self.n_id,
                                     kwargs.get('g_leak', 0.8))
        self.e_leak = self.dae.add_c('e_leak' + self.n_id,
                                     kwargs.get('e_leak', -70.0))

        #: Parameters of Isyn
        self.g_syn = self.dae.add_c('g_syn' + self.n_id,
                                    kwargs.get('g_syn', 0.1))
        self.e_syn = self.dae.add_c('e_syn' + self.n_id,
                                    kwargs.get('e_syn', 0.0))
        self.v_hs = self.dae.add_c('v_hs' + self.n_id,
                                   kwargs.get('v_hs', -43.0))
        self.gamma_s = self.dae.add_c('gamma_s' + self.n_id,
                                      kwargs.get('gamma_s', -0.42))

        #: Parameters of Iapp
        self.g_app = self.dae.add_c('g_app' + self.n_id,
                                    kwargs.get('g_app', 0.19))
        self.e_app = self.dae.add_c('e_app' + self.n_id,
                                    kwargs.get('e_app', 0.0))

        #: Other constants
        self.c_m = self.dae.add_c('c_m' + self.n_id,
                                  kwargs.get('c_m', 1.0))

        #: State Variables
        #: pylint: disable=invalid-name
        #: Membrane potential
        self.v = self.dae.add_x('V_' + self.n_id,
                                kwargs.get('v0', 0.0))
        self.m_na = self.dae.add_x(
            'm_na_' + self.n_id, kwargs.get('m_na0', 0.0))
        self.h_na = self.dae.add_x(
            'h_na_' + self.n_id, kwargs.get('h_na0', 0.0))
        self.m_k = self.dae.add_x('m_k_' + self.n_id,
                                  kwargs.get('m_k0', 0.0))
        self.m_q = self.dae.add_x('m_q_' + self.n_id,
                                  kwargs.get('m_q0', 0.0))

        #: ODE
        self.vdot = self.dae.add_ode('vdot_' + self.n_id, 0.0)
        self.m_na_dot = self.dae.add_ode('m_na_dot_' + self.n_id, 0.0)
        self.h_na_dot = self.dae.add_ode('h_na_dot_' + self.n_id, 0.0)
        self.m_k_dot = self.dae.add_ode('m_k_dot_' + self.n_id, 0.0)
        self.m_q_dot = self.dae.add_ode('m_q_dot_' + self.n_id, 0.0)

        #: External Input
        self.g_app = self.dae.add_u('g_app_' + self.n_id,
                                    kwargs.get('g_app', 0.0))
        self.e_app = self.dae.add_u('e_app_' + self.n_id,
                                    kwargs.get('e_app', 0.0))

        #: Add outputs
        if kwargs.get('output'):
            self.dae.add_y(self.v)

        #: ODE
        self.ode_rhs()

    def add_ode_input(self, neuron, **kwargs):
        """ Add relevant external inputs to the ode.
        Parameters
        ----------
        neuron : <LIF_Danner>
            Neuron model from which the input is received.
        weight : <float>
            Strength of the synapse between the two neurons
        """

        g_syn = self.dae.add_p('g_syn_' + self.n_id,
                               kwargs.get('g_syn', 0.0))
        e_syn = self.dae.add_p('e_syn_' + self.n_id,
                               kwargs.get('e_syn', 0.0))
        gamma_s = self.dae.add_p('gamma_s_' + self.n_id,
                                 kwargs.get('gamma_s', 0.0))
        v_h_s = self.dae.add_p('v_h_s_' + self.n_id,
                               kwargs.get('v_h_s', 0.0))

        s_inf = cas.inv(
            1. + cas.exp(gamma_s.sym*(
                neuron.neuron_out() - v_h_s.sym)))

        self.vdot.sym += -(g_syn.sym*s_inf *
                           (self.v.sym - e_syn.sym))/self.c_m.sym
        return

    def ode_rhs(self):
        """Generate initial ode rhs."""

        #: alpha_m_Na(V)
        a_m_nap = (self.am1_nap.sym * (self.am2_nap.sym - self.v.sym)) / (
            cas.exp(self.am3_nap.sym * (self.am2_nap.sym - self.v.sym)) - 1)

        #: beta_m_Na(V)
        b_m_nap = (self.bm1_nap.sym * (self.bm2_nap.sym - self.v.sym)) / (
            cas.exp(self.bm3_nap.sym * (self.bm2_nap.sym - self.v.sym)) - 1)

        #: alpha_m_Na(V)
        a_h_nap = self.ah1_nap.sym * cas.exp(
            self.ah3_nap.sym * (self.ah2_nap.sym - self.v.sym))

        #: beta_m_Na(V)
        b_h_nap = (self.bh1_nap.sym) / (
            cas.exp(self.bh3_nap.sym * (self.bh2_nap.sym - self.v.sym)) + 1)

        #: m_na_dot
        self.m_na_dot.sym = a_m_nap*(
            1 - self.m_na.sym) - b_m_nap*self.m_na.sym

        #: h_na_dot
        self.h_na_dot.sym = a_h_nap*(1 - self.h_na.sym) - b_h_nap*self.h_na.sym

        #: Inap
        #: pylint: disable=no-member
        i_nap = self.g_nap.sym * self.m_na.sym * self.h_na.sym * (
            self.v.sym - self.e_nap.sym)

        #: alpha_m_K
        a_m_k = (self.am1_k.sym * (self.am2_k.sym - self.v.sym)) / (
            cas.exp(self.am3_k.sym * (self.am2_k.sym - self.v.sym)) - 1)

        #: beta_m_K
        b_m_k = self.bm1_k.sym * cas.exp(
            self.bm3_k.sym * (self.bm2_k.sym - self.v.sym))

        #: m_k_dot
        self.m_k_dot.sym = a_m_k*(1 - self.m_k.sym) - b_m_k*self.m_k.sym

        #: Ik
        #: pylint: disable=no-member
        i_k = self.g_k.sym * self.m_k.sym * (self.v.sym - self.e_k.sym)

        #: m_q_inf
        m_q_inf = cas.inv(1 + cas.exp(
            self.gamma_q.sym * (self.v.sym - self.v_m_q.sym)))

        #: alpha_m_q
        a_m_q = m_q_inf * self.r_q.sym

        #: beta_m_q
        b_m_q = (1 - m_q_inf) * self.r_q.sym

        #: m_q_dot
        self.m_q_dot.sym = a_m_q * (1 - self.m_q.sym) - b_m_q * self.m_q.sym

        #: Iq
        #: pylint: disable=no-member
        i_q = self.g_q.sym * self.m_q_dot.sym * (self.v.sym - self.e_q.sym)

        #: Ileak
        i_leak = self.g_leak.sym * (self.v.sym - self.e_leak.sym)

        #: Iapp
        i_app = self.g_app.sym * (self.v.sym - self.e_app.sym)

        #: dV
        self.vdot.sym = -(
            i_nap + i_k + i_q + i_leak + i_app)/self.c_m.sym
        return

    def ode_alg_eqn(self):
        """ ODE Algebraic Variables."""
        pass

    def neuron_out(self):
        """Neuron activation function.
        Parameters
        ----------
        m_potential: float
            Neuron membrane potential
        """
        return self.v.sym


class ConstantAndInhibit(Neuron):

    def __init__(self, n_id, dae, **kwargs):
        #: Neuron ID
        self.n_id = n_id
        self.dae = dae

        # Initialize parameters
        self.K = self.dae.add_c('K_' + self.n_id, kwargs.get('K', -500.))
        self.B = self.dae.add_c('B_' + self.n_id, kwargs.get('B', -20.))
        self.ades = self.dae.add_p(
            'ades_' + self.n_id, kwargs.get('ades', 0.4))

        #: Initialize states
        self.a = self.dae.add_x('a_' + self.n_id, kwargs.get('a0', 0.4))
        self.adot = self.dae.add_x(
            'adot_' + self.n_id, kwargs.get('adot0', 0.0))

        #: External inputs
        self.a_inh = self.dae.add_u('ext_in_' + self.n_id)

        #: ODE RHS
        self.da = self.dae.add_ode('da_' + self.n_id, 0.0)
        self.dda = self.dae.add_ode('dda_' + self.n_id, 0.0)

        # ODE
        self.ode_rhs()

    def ode_rhs(self):
        """ Generate the ODE. Internal Setup Function."""
        self.da.sym = self.adot.sym
        self.dda.sym = self.K.sym * \
            (self.a.sym - self.ades.sym) + self.B.sym * self.adot.sym

    def ode_alg_eqn(self):
        """ Abstract class. """
        pass

    def add_ode_input(self, neuron, **kwargs):
        """ Add relevant external inputs to the ode."""

        K_f = self.dae.add_p('K_f_' + neuron.n_id +
                             '_to_' + self.n_id, kwargs.get('K_f', 100.))

        self.dda.sym -= K_f.sym * neuron.m.sym  # neuron.neuron_out()
        return
