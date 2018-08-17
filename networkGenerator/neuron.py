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


class LIF_Danner_Nap(Neuron):
    """Leaky Integrate and Fire Neuron Based on Danner et.al.

    """

    def __init__(self, n_id, **kwargs):
        super(
            LIF_Danner_Nap, self).__init__(neuron_type='lif_danner_nap')

        self.n_id = n_id  #: Unique neuron identifier

        #: Constants
        self.c_m = 10.0  #: pF

        self.g_nap = 4.5  #: nS
        self.e_na = 50.0  #: mV

        self.v1_2_m = -40.0  #: mV
        self.k_m = -6.0  #: mV

        self.v1_2_h = -45.0  #: mV
        self.k_h = 4.0  #: mV

        self.v1_2_t = -35.0  #: mV
        self.k_t = 15.0  #: mV

        self.g_leak = 4.5  #: nS
        self.e_leak = -62.5  #: mV

        self.tau_0 = 80.0  #: ms
        self.tau_max = 160.0  #: ms
        self.tau_noise = 10.0  #: ms

        self.v_max = 0.0  #: mV
        self.v_thr = -50.0  #: mV

        self.g_syn_e = 10.0  #: nS
        self.g_syn_i = 10.0  #: nS
        self.e_syn_e = -10.0  #: mV
        self.e_syn_i = -75.0  #: mV

        #: State Variables
        self.v = cas.SX.sym('V_' + self.n_id)  #: Membrane potential
        self.h = cas.SX.sym('h_' + self.n_id)
        self.i_noise = cas.SX.sym('In_' + self.n_id)

        #: ODE
        self.vdot = None
        self.hdot = None

        #: External Input (BrainStem Drive)
        self.alpha = cas.SX.sym('alpha_' + self.n_id)
        self.m_e = kwargs.pop('m_e', 0.0)  #: m_E,i
        self.m_i = kwargs.pop('m_i', 0.0)  #: m_I,i
        self.b_e = kwargs.pop('b_e', 0.0)  #: m_E,i
        self.b_i = kwargs.pop('b_i', 0.0)  #: m_I,i

        self.d_e = self.m_e * self.alpha + self.b_e
        self.d_i = self.m_i * self.alpha + self.b_i

        self.sum_syn_e = 0.0
        self.sum_syn_i = 0.0

        return

    def ode_add_input(self, neuron, weight, **kwargs):
        """ Add relevant external inputs to the ode.
        Parameters
        ----------
        neuron : <LIF_Danner>
            Neuron model from which the input is received.
        weight : <float>
            Strength of the synapse between the two neurons
        """
        #: Weight squashing function
        def s_w(w): return w * (w >= 0.0)
        if np.sign(weight) == 1:
            #: Excitatory Synapse
            biolog.debug('Adding excitatory signal of weight {}'.format(
                s_w(weight)))
            self.sum_syn_e += s_w(weight) * neuron.neuron_out()
        elif np.sign(weight) == -1:
            #: Inhibitory Synapse
            biolog.debug('Adding inhibitory signal of weight {}'.format(
                s_w(-weight)))
            self.sum_syn_i += s_w(-weight) * neuron.neuron_out()

        return

    def _ode_rhs(self):
        """Generate initial ode rhs."""

        #: tau_h(V)
        tau_h = self.tau_0 + (self.tau_max - self.tau_0) / \
            cas.cosh((self.v - self.v1_2_t) / self.k_t)

        #: h_inf(V)
        h_inf = cas.inv(1.0 + cas.exp((self.v - self.v1_2_h) / self.k_h))

        #: Slow inactivation
        self.hdot = (h_inf - self.h) / tau_h

        #: m(V)
        m = cas.inv(1.0 + cas.exp((self.v - self.v1_2_m) / self.k_m))

        #: Inap
        #: pylint: disable=no-member
        i_nap = self.g_nap * m * self.h * (self.v - self.e_na)

        #: Ileak
        i_leak = self.g_leak * (self.v - self.e_leak)

        #: ISyn_Excitatory
        i_syn_e = self.g_syn_e * (self.sum_syn_e + self.d_e) * (
            self.v - self.e_syn_e)

        #: ISyn_Inhibitory
        i_syn_i = self.g_syn_i * (self.sum_syn_i + self.d_i) * (
            self.v - self.e_syn_i)

        #: dV
        self.vdot = - i_nap - i_leak - i_syn_e - i_syn_i
        return

    def ode_rhs(self):
        """ ODE RHS."""
        self._ode_rhs()
        return [self.vdot / self.c_m, self.hdot]

    def ode_states(self):
        """ ODE States."""
        return [self.v, self.h]

    def ode_params(self):
        """ Generate neuron parameters."""
        return [self.alpha]

    def neuron_out(self):
        """ Output of the neuron model."""
        _cond = cas.logic_and(self.v_thr <= self.v, self.v < self.v_max)
        _f = (self.v - self.v_thr) / (self.v_max - self.v_thr)
        return cas.if_else(_cond, _f, 1.) * (self.v > self.v_thr)


class LIF_Danner(Neuron):
    """Leaky Integrate and Fire Neuron Based on Danner et.al.

    """

    def __init__(self, n_id, **kwargs):
        super(LIF_Danner, self).__init__(neuron_type='lif_danner')
        self.n_id = n_id

        self.n_id = n_id  #: Unique neuron identifier

        #: Constants
        self.c_m = 10.0  #: pF

        self.v1_2_m = -40.0  #: mV
        self.k_m = -6.0  #: mV

        self.v1_2_h = -45.0  #: mV
        self.k_h = -4.0  #: mV

        self.v1_2_t = -35.0  #: mV
        self.k_t = -15.0  #: mV

        self.g_leak = 2.8  #: nS
        self.e_leak = -60.0  #: mV

        self.tau_noise = 10.0  #: ms

        self.v_max = 0.0  #: mV
        self.v_thr = -50.0  #: mV

        self.g_syn_e = 10.0  #: nS
        self.g_syn_i = 10.0  #: nS
        self.e_syn_e = -10.0  #: mV
        self.e_syn_i = -75.0  #: mV

        #: State Variables
        self.v = cas.SX.sym('V_' + self.n_id)  #: Membrane potential
        self.h = cas.SX.sym('h_' + self.n_id)
        self.i_noise = cas.SX.sym('In_' + self.n_id)

        #: ODE
        self.vdot = None
        self.hdot = None

        #: External Input (BrainStem Drive)
        self.alpha = cas.SX.sym('alpha_' + self.n_id)
        self.m_e = kwargs.pop('m_e', 0.0)  #: m_E,i
        self.m_i = kwargs.pop('m_i', 0.0)  #: m_I,i
        self.b_e = kwargs.pop('b_e', 0.0)  #: m_E,i
        self.b_i = kwargs.pop('b_i', 0.0)  #: m_I,i

        self.d_e = self.m_e * self.alpha + self.b_e
        self.d_i = self.m_i * self.alpha + self.b_i

        self.sum_syn_e = 0.0
        self.sum_syn_i = 0.0

        return

    def ode_add_input(self, neuron, weight, **kwargs):
        """ Add relevant external inputs to the ode.
        Parameters
        ----------
        neuron : <LIF_Danner>
            Neuron model from which the input is received.
        weight : <float>
            Strength of the synapse between the two neurons
        """
        #: Weight squashing function
        def s_w(w): return w * (w >= 0.0)
        if np.sign(weight) == 1:
            #: Excitatory Synapse
            biolog.debug('Adding excitatory signal of weight {}'.format(
                s_w(weight)))
            self.sum_syn_e += s_w(weight) * neuron.neuron_out()
        elif np.sign(weight) == -1:
            #: Inhibitory Synapse
            biolog.debug('Adding inhibitory signal of weight {}'.format(
                s_w(-weight)))
            self.sum_syn_i += s_w(-weight) * neuron.neuron_out()

        return

    def _ode_rhs(self):
        """Generate initial ode rhs."""

        #: Ileak
        i_leak = self.g_leak * (self.v - self.e_leak)

        #: ISyn_Excitatory
        i_syn_e = self.g_syn_e * (self.sum_syn_e + self.d_e) * (
            self.v - self.e_syn_e)

        #: ISyn_Inhibitory
        i_syn_i = self.g_syn_i * (self.sum_syn_i + self.d_i) * (
            self.v - self.e_syn_i)

        #: dV
        self.vdot = - i_leak - i_syn_e - i_syn_i
        return

    def ode_rhs(self):
        """ ODE RHS."""
        self._ode_rhs()
        return [self.vdot / self.c_m]

    def ode_states(self):
        """ ODE States."""
        return [self.v]

    def ode_params(self):
        """ Generate neuron parameters."""
        return [self.alpha]

    def neuron_out(self):
        """ Output of the neuron model."""
        _cond = cas.logic_and(self.v_thr <= self.v, self.v < self.v_max)
        _f = (self.v - self.v_thr) / (self.v_max - self.v_thr)
        return cas.if_else(_cond, _f, 1.) * (self.v > self.v_thr)


class IntegrateAndFire(Neuron):
    """Integrate & Fire Neuron Model."""

    def __init__(self, n_id, **kwargs):
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

        #: Initialize parameters
        self.tau = kwargs.pop('tau', 0.1)
        self.bias = kwargs.pop('bias', -2.75)
        #: pylint: disable=invalid-name
        self.D = kwargs.pop('D', 1.0)

        #: Initialize states
        self.m = cas.SX.sym('m_' + self.n_id)

        #: External inputs
        self.ext_in = cas.SX.sym('ext_in_' + self.n_id)
        self.ext_sum = 0.0

        self.mdot = None

    def ode_add_input(self, neuron, weight):
        """ Add relevant external inputs to the ode."""
        self.ext_sum += neuron.neuron_output()*weight
        return

    def _ode_rhs(self):
        """ Generate the ODE. Internal Setup Function."""
        self.mdot = -self.m + self.ext_sum + self.ext_in

    def ode_rhs(self):
        """ Generate ODE RHS."""
        self._ode_rhs()
        return [self.mdot / self.tau]

    def ode_states(self):
        """ ODE States."""
        return [self.m]

    def ode_params(self):
        """ Generate neuron parameters."""
        return [self.ext_in]

    def neuron_output(self):
        """Neuron activation function.
        Parameters
        ----------
        m_potential: float
            Neuron membrane potential
        """
        exp = np.exp  # pylint: disable = no-member
        return 1. / (1. + exp(-self.D * (
            self.m + self.bias)))


class LIF_Daun_Interneuron(Neuron):
    """Leaky Integrate and Fire Interneuron.
    Based on Silvia Daun and Tbor's model.
    """

    def __init__(self, n_id, **kwargs):
        super(LIF_Daun_Interneuron, self).__init__(
            neuron_type='lif_daun_interneuron')

        self.n_id = n_id  #: Unique neuron identifier

        #: Constants
        self.g_nap = kwargs.get('g_nap', 10.0)
        self.e_nap = kwargs.get('e_nap', 50.0)

        #: Parameters of h
        self.v_h_h = kwargs.get('v_h_h', -30.0)
        self.gamma_h = kwargs.get('gamma_h', 0.1667)

        #: Parameters of tau
        self.v_t_h = kwargs.get('v_t_h', -30.0)
        self.eps = kwargs.get('eps', 0.0023)
        self.gamma_t = kwargs.get('gamma_t', 0.0833)

        #: Parameters of m
        self.v_h_m = kwargs.get('v_h_m', -37.0)
        self.gamma_m = kwargs.get('gamma_m', -0.1667)

        #: Parameters of Ileak
        self.g_leak = kwargs.get('g_leak', 2.8)
        self.e_leak = kwargs.get('e_leak', -65.0)

        #: Other constants
        self.c_m = kwargs.get('c_m', 0.9154)

        #: Sum of external neuron connections
        self.i_syn = 0.0

        #: State Variables
        #: pylint: disable=invalid-name
        self.v = cas.SX.sym('V_' + self.n_id)  #: Membrane potential
        self.h = cas.SX.sym('h_' + self.n_id)

        #: ODE
        self.vdot = None
        self.hdot = None

        self.z_i_syn = cas.SX.sym('z_i_syn_' + self.n_id)

        #: External Input
        self.g_app = cas.SX.sym('g_app_' + self.n_id)
        self.e_app = cas.SX.sym('e_app_' + self.n_id)

        return

    def ode_add_input(self, neuron, _, **kwargs):
        """ Add relevant external inputs to the ode.
        Parameters
        ----------
        neuron : <LIF_Danner>
            Neuron model from which the input is received.
        weight : <float>
            Strength of the synapse between the two neurons"""

        g_syn = kwargs.pop('g_syn', 0.0)
        e_syn = kwargs.pop('e_syn', 0.0)
        gamma_s = kwargs.pop('gamma_s', 0.0)
        v_h_s = kwargs.pop('v_h_s', 0.0)

        s_inf = cas.inv(
            1. + cas.exp(gamma_s*(neuron.neuron_out() - v_h_s)))

        self.i_syn += g_syn*s_inf*(self.v - e_syn)
        return

    def _ode_rhs(self):
        """Generate initial ode rhs."""

        #: tau_h(V)
        tau_h = cas.inv(
            self.eps*cas.cosh(self.gamma_t*(self.v - self.v_t_h)))

        #: h_inf(V)
        h_inf = cas.inv(
            1. + cas.exp(self.gamma_h*(self.v - self.v_h_h)))

        #: Slow inactivation
        self.hdot = (h_inf - self.h)/tau_h

        #: m_inf(V)
        m_inf = cas.inv(
            1. + cas.exp(self.gamma_m*(self.v - self.v_h_m)))

        #: Inap
        #: pylint: disable=no-member
        i_nap = self.g_nap * m_inf * self.h * (self.v - self.e_nap)

        #: Ileak
        i_leak = self.g_leak * (self.v - self.e_leak)

        #: Iapp
        i_app = self.g_app * (self.v - self.e_app)

        #: dV
        self.vdot = -(i_nap + i_leak + i_app + self.i_syn)
        return

    def ode_rhs(self):
        """ ODE RHS."""
        self._ode_rhs()
        return [self.vdot / self.c_m, self.hdot]

    def ode_states(self):
        """ ODE States."""
        return [self.v, self.h]

    def ode_params(self):
        """ Generate neuron parameters."""
        return [self.g_app, self.e_app]

    def ode_alg_var(self):
        """ ODE Algebraic Variables."""
        return [self.z_i_syn]

    def ode_alg_eqn(self):
        """ ODE Algebraic Variables."""
        return [self.z_i_syn - self.i_syn]

    def neuron_out(self):
        """Neuron activation function.
        Parameters
        ----------
        m_potential: float
            Neuron membrane potential
        """
        return self.v


class LIF_Motorneuron(Neuron):
    """Leaky Integrate and Fire Motorneuron.
    """

    def __init__(self):
        super(LIF_Motorneuron, self).__init__(
            neuron_type='lif_motoneuron')
        #: Neuron constants
        #: Parameters of INaP
        self.g_nap = 10.0
        self.e_nap = 55.0
        self.am1_nap = 0.32
        self.am2_nap = -51.90
        self.am3_nap = 0.25
        self.bm1_nap = -0.280
        self.bm2_nap = -24.90
        self.bm3_nap = -0.2
        self.ah1_nap = 0.1280
        self.ah2_nap = -48.0
        self.ah3_nap = 0.0556
        self.bh1_nap = 4.0
        self.bh2_nap = -25.0
        self.bh3_nap = 0.20

        #: Parameters of IK
        self.g_k = 2.0
        self.e_k = -80.0
        self.am1_k = 0.0160
        self.am2_k = -29.90
        self.am3_k = 0.20
        self.bm1_k = 0.250
        self.bm2_k = -45.0
        self.bm3_k = 0.025

        #: Parameters of Iq
        self.g_q = 12.0
        self.e_q = -80.0
        self.vhm_q = -30.0
        self.gamma_q = -0.6
        self.r_q = 0.0005

        #: Parameters of Ileak
        self.g_leak = 0.8
        self.e_leak = -70.0

        #: Parameters of Isyn
        self.g_syn = 0.1
        self.e_syn = 0.0
        self.v_hs = -43.0
        self.gamma_s = -0.42

        #: Parameters of Iapp
        self.g_app = 0.19
        self.e_app = 0.0

        #: Other constants
        self.c_m = 1.0

    def ode(self, time, state, v_syn):
        """
        Parameters
        ----------
        time: < float >
            Current simulation time step
        states: < np.array >
            Neuron states
                * V: Membrane potential
                * h: Inactivation variable
        v_syn: < np.array >
            External synaptic membrane potentials
        Returns
        -------
        out: < np.array >
            Rate of change of neuron membrane potential

        """
        _V = state[0]  #: Membrane potential
        _m_nap = state[1]  #: Inactivation variable
        _h_nap = state[2]  #: Inactivation variable
        _m_k = state[3]  #: Inactivation variable
        _m_q = state[4]  #: Inactivation variable

        #: Helper functions
        def ix_func(g_x, e_x, m_x, h_x=1.):
            """ Compute the current."""
            p = 1
            return g_x * (m_x**p) * h_x * (_V - e_x)

        def inact_func(y, alpha_y, beta_y):
            """
            Compute the differential of inactivating functions
            """
            return alpha_y * (1 - y) - beta_y * y

        def v_func(gam, v_h, v_ext=None):
            if v_ext is None:
                return 1. / (1 + np.exp(gam * (_V - v_h)))
            else:
                return 1. / (1 + np.exp(gam * (v_ext - v_h)))

        #: Inap
        alpha_m_nap = (self.am1_nap * (self.am2_nap - _V)) / (
            np.exp(self.am3_nap * (self.am2_nap - _V)) - 1)

        beta_m_nap = (self.bm1_nap * (self.bm2_nap - _V)) / (
            np.exp(self.bm3_nap * (self.bm2_nap - _V)) - 1)

        alpha_h_nap = self.ah1_nap * np.exp(self.ah3_nap * (self.ah2_nap - _V))

        beta_h_nap = (self.bh1_nap) / (
            np.exp(self.bh3_nap * (self.bh2_nap - _V)) + 1)

        I_nap = ix_func(self.g_nap, self.e_nap, _m_nap, _h_nap)

        #: Ik
        alpha_m_k = (self.am1_k * (self.am2_k - _V)) / (
            np.exp(self.am3_k * (self.am2_k - _V)) - 1)

        beta_m_k = self.bm1_k * np.exp(
            self.bm3_k * (self.bm2_k - _V))

        I_k = ix_func(self.g_k, self.e_k, _m_k)

        #: Iq
        # TODO: Check the equation : Sq
        m_q_inf = 1. / (1 + np.exp(self.gamma_q * (_V - self.vhm_q)))

        alpha_m_q = m_q_inf * self.r_q

        beta_m_q = (1 - m_q_inf) * self.r_q

        I_q = ix_func(self.g_q, self.e_q, _m_q)

        #: Ileak
        I_leak = self.g_leak * (_V - self.e_leak)

        #: Iapp
        I_app = self.g_app * (_V - self.e_app)

        #: Isyn
        I_syn = np.sum(self.g_syn * v_func(
            self.gamma_s, self.v_hs, v_syn) * (_V - self.e_syn))

        #: dV
        _dV = -(I_nap + I_k + I_q + I_leak + I_syn + I_app) / self.c_m

        _dm_nap = inact_func(_m_nap, alpha_m_nap, beta_m_nap)

        _dh_nap = inact_func(_h_nap, alpha_h_nap, beta_h_nap)

        _dm_k = inact_func(_m_k, alpha_m_k, beta_m_k)

        _dm_q = inact_func(_m_q, alpha_m_q, beta_m_q)

        return np.array([_dV, _dm_nap, _dh_nap, _dm_k, _dm_q])


def main():
    neuron1 = LIF_Interneuron()
    biolog.debug('Neuron type : {}'.format(neuron1.neuron_type))
    biolog.debug('Neuron ode out : {}'.format(
        neuron1.ode(0.0, np.array([-56., 0.0]), np.arange(0, -5, 1))))

    neuron2 = LIF_Motorneuron()
    biolog.debug('Neuron type : {}'.format(neuron2.neuron_type))
    biolog.debug('Neuron ode out : {}'.format(
        neuron2.ode(0.0, np.array([-56., 0.0, 0.0, 0.0, 0.0]), np.arange(
            0, -5, 1))))


if __name__ == '__main__':
    main()
