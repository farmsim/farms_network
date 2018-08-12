"""Implementation of different neuron models."""
import biolog
import numpy as np
import casadi as cas
import casadi.tools as cast


class Neuron(object):
    """Base neuron class.
    Inherits from Casadi Dae Builder Class.
    """

    def __init__(self, neuron_type, is_ext):
        super(Neuron, self).__init__()
        self._neuron_type = neuron_type  # : Type of neuron
        self.is_ext = is_ext  #: Is external input

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
    def __init__(self, n_id, neuron_type, is_ext=True):
        super(LIF_Danner_Nap, self).__init__(neuron_type, is_ext)
        self.n_id = n_id
        self.neuron_type = neuron_type
        self.is_ext = is_ext

        self.n_id = n_id  #: Unique neuron identifier

        #: Constants
        self.c_m = 10  #: pF

        self.g_nap = 4.5  #: nS
        self.e_na = 50  #: mV

        self.v1_2_m = -40  #: mV
        self.k_m = -6  #: mV

        self.v1_2_h = -45  #: mV
        self.k_h = -4  #: mV

        self.v1_2_t = -35  #: mV
        self.k_t = -15  #: mV

        self.g_leak = 2.8  #: nS
        self.e_leak = -60  #: mV

        self.tau_0 = 80  #: ms
        self.tau_max = 160  #: ms
        self.tau_noise = 10  #: ms

        self.v_max = 0  #: mV
        self.v_thr = -50  #: mV

        self.g_syn_e = 10.  #: nS
        self.g_syn_i = 10.  #: nS
        self.e_syn_e = -10.  #: mV
        self.e_syn_i = -75.  #: mV

        #: State Variables
        self.v = cas.SX.sym('V_' + self.n_id)  #: Membrane potential
        self.h = cas.SX.sym('h_' + self.n_id)
        self.i_noise = cas.SX.sym('In_' + self.n_id)

        #: ODE
        self.vdot = None
        self.hdot = None

        #: External Input (BrainStem Drive)
        self.alpha = cas.SX.sym('alpha_' + self.n_id)
        self.m_e_i = 0.0  #: m_E,i
        self.m_i_i = 0.0  #: m_I,i
        self.b_e_i = 0.0  #: m_E,i
        self.b_i_i = 0.0  #: m_I,i

        self.d_e_i = self.m_e_i*self.alpha + self.b_e_i
        self.d_i_i = self.m_i_i*self.alpha + self.b_i_i

        self.sum_i_syn_e = 0.0
        self.sum_i_syn_i = 0.0

        #: Weight squashing function
        _x = cas.SX.sym('x')
        self.s_x = cas.Function('s_x',
                                [_x], [_x*(_x>=0.0)])
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
        if np.sign(weight) == 1:
            #: Excitatory Synapse
            self.sum_i_syn_e += self.s_x(weight)*neuron.neuron_out()
        elif np.sign(weight) == -11:
            #: Excitatory Synapse
            self.sum_i_syn_i += self.s_x(weight)*neuron.neuron_out()

        return

    def _ode_rhs(self):
        """Generate initial ode rhs."""

        #: tau_h(V)
        tau_h = self.tau_0 + (
            self.tau_max - self.tau_0)/cas.cosh(
                (self.v - self.v1_2_t)/self.k_t)

        #: h_inf(V)
        h_inf = cas.inv(1 + cas.exp(self.v - self.v1_2_h)/self.k_h)

        #: m(V)
        m = cas.inv(1 + cas.exp(self.v - self.v1_2_m)/self.k_m)

        #: Slow inactivation
        self.hdot = (h_inf - self.h)/tau_h

        #: Inap
        #: pylint: disable=no-member
        i_nap = self.g_nap*m*self.h*(self.v - self.e_na)

        #: Ileak
        i_leak = self.g_leak*(self.v - self.e_leak)

        #: ISyn_Excitatory
        i_syn_e = self.g_syn_e*(self.sum_i_syn_e + self.d_e_i)*(
            self.v - self.e_syn_e)

        #: ISyn_Inhibitory
        i_syn_i = self.g_syn_i*(self.sum_i_syn_i + self.d_i_i)*(
            self.v - self.e_syn_i)

        #: dV
        self.vdot = - i_nap - i_leak - i_syn_e - i_syn_i
        return

    def ode_rhs(self):
        """ ODE RHS."""
        self._ode_rhs()
        return [self.vdot/self.c_m, self.hdot]

    def ode_states(self):
        """ ODE States."""
        return [self.v, self.h]

    def ode_params(self):
        """ Generate neuron parameters."""
        return [self.alpha]

    def neuron_out(self):
        """ Output of the neuron model."""
        _cond = cas.logic_and(self.v_thr <= self.v, self.v < self.v_max)
        _f = (self.v - self.v_thr)/(self.v_thr - self.v_max)
        return cas.if_else(_cond, _f, 1.)*(self.v < self.v_thr)


class IntegrateAndFire(Neuron):
    """Integrate & Fire Neuron Model."""

    def __init__(self, n_id, neuron_type=None, is_ext=True,
                 tau=0.1, bias=-2.75, D=1.0):
        """Initialize.

        Parameters
        ----------
        id: str
            Unique ID for the neuron in the network.
        """
        super(IntegrateAndFire, self).__init__(
            neuron_type=neuron_type, is_ext=is_ext)

        #: Neuron ID
        self.n_id = n_id

        #: Initialize parameters
        self.tau = tau
        self.bias = bias
        #: pylint: disable=invalid-name
        self.D = D

        #: Initialize states
        self.states = cast.struct_symSX(['m_' + self.n_id])
        [self.m] = self.states[...]

        #: External inputs
        self.ext_in = cas.SX.sym('ext_in_' + self.n_id)

        if self.is_ext:
            self.mdot = (-self.m + self.ext_in)
        else:
            self.mdot = (-self.m)

    def ode_add_input(self, in_):
        """ Add relevant external inputs to the ode."""
        self.mdot += in_
        return

    def ode_rhs(self):
        """ Generate ODE RHS."""
        return [self.mdot/self.tau]

    def ode_states(self):
        """ ODE States."""
        return [self.states.cat]

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


class Leaky_Interneuron(Neuron):
    """Leaky Integrate and Fire Interneuron.
    """

    def __init__(self, n_id, is_ext=True):
        super(Leaky_Interneuron, self).__init__(
            neuron_type='leaky_interneuron', is_ext=is_ext)

        self.n_id = n_id  #: Unique neuron identifier
        #: Constants
        self.g_nap = 10.0
        self.e_nap = 50.0
        self.v_hm = -37.0
        self.gamma_m = -0.1667
        self.eps = 0.002
        #: Parameters of h
        self.v_hh = -30.0
        self.gamma_h = 0.1667
        self.v_htau = -30.0
        self.gamma_tau = 0.0833
        #: Parameters of Ileak
        self.g_leak = 2.8
        self.e_leak = -65.0

        #: Parameters of Isyn
        self.e_syn = 0.0
        self.v_hs = -43.0
        self.gamma_s = -0.42
        #: Parameters of Iapp
        self.g_app = 0.16
        self.e_app = 0.0
        #: Other constants
        self.c_m = 1.0

        #: State Variables
        self.V = cas.SX.sym('V_' + self.n_id)  #: Membrane potential
        self.h = cas.SX.sym('h' + self.n_id)

        #: ODE
        self.vdot = None
        self.hdot = None
        self._ode_rhs()

        #: External Input
        self.ext_in = cas.SX.sym('ext_in_' + self.n_id)

        return

    def ode_add_input(self, v_syn, g_syn=0.0, gamma=0.0, v_h=0.0):
        """ Add relevant external inputs to the ode."""

        #: Isyn
        #: pylint: disable=no-member
        I_syn_1 = 1. / (1. + np.exp(gamma*(v_syn - v_h)))
        I_syn = g_syn*I_syn_1*(self.V - self.e_syn)
        self.vdot += I_syn
        return

    def _ode_rhs(self):
        """Generate initial ode rhs."""
        #: Inap
        #: pylint: disable=no-member
        I_nap_1 = 1. / (1 + np.exp(self.gamma_m*(self.V - self.v_hm)))
        I_nap = self.g_nap*I_nap_1*self.h*(self.V - self.e_nap)

        #: Ileak
        I_leak = self.g_leak*(self.V - self.e_leak)

        #: Iapp
        I_app = self.g_app*(self.V - self.e_app)

        #: hinf
        h_inf = 1. / (1 + np.exp(self.gamma_h*(self.V - self.v_hh)))

        #: tau_h
        tau_h = 1. / (
            self.eps*(np.cosh(self.gamma_tau*(self.V - self.v_htau))))

        #: dV
        self.vdot = I_nap + I_leak + I_app

        #: dh
        self.hdot = (h_inf - self.h)/tau_h
        return

    def ode_rhs(self):
        """ ODE RHS."""
        return [-self.vdot/self.c_m, self.hdot]

    def ode_states(self):
        """ ODE States."""
        return [self.V, self.h]

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
        pass


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
            return g_x*(m_x**p)*h_x*(_V - e_x)

        def inact_func(y, alpha_y, beta_y):
            """
            Compute the differential of inactivating functions
            """
            return alpha_y*(1-y) - beta_y*y

        def v_func(gam, v_h, v_ext=None):
            if v_ext is None:
                return 1. / (1 + np.exp(gam*(_V - v_h)))
            else:
                return 1. / (1 + np.exp(gam*(v_ext - v_h)))

        #: Inap
        alpha_m_nap = (self.am1_nap*(self.am2_nap - _V))/(
            np.exp(self.am3_nap*(self.am2_nap - _V)) - 1)

        beta_m_nap = (self.bm1_nap*(self.bm2_nap - _V))/(
            np.exp(self.bm3_nap*(self.bm2_nap - _V)) - 1)

        alpha_h_nap = self.ah1_nap*np.exp(self.ah3_nap*(self.ah2_nap - _V))

        beta_h_nap = (self.bh1_nap)/(
            np.exp(self.bh3_nap*(self.bh2_nap - _V)) + 1)

        I_nap = ix_func(self.g_nap, self.e_nap, _m_nap, _h_nap)

        #: Ik
        alpha_m_k = (self.am1_k*(self.am2_k - _V))/(
            np.exp(self.am3_k*(self.am2_k - _V)) - 1)

        beta_m_k = self.bm1_k*np.exp(
            self.bm3_k*(self.bm2_k - _V))

        I_k = ix_func(self.g_k, self.e_k, _m_k)

        #: Iq
        # TODO: Check the equation : Sq
        m_q_inf = 1./(1 + np.exp(self.gamma_q*(_V - self.vhm_q)))

        alpha_m_q = m_q_inf*self.r_q

        beta_m_q = (1 - m_q_inf)*self.r_q

        I_q = ix_func(self.g_q, self.e_q, _m_q)

        #: Ileak
        I_leak = self.g_leak*(_V - self.e_leak)

        #: Iapp
        I_app = self.g_app*(_V - self.e_app)

        #: Isyn
        I_syn = np.sum(self.g_syn*v_func(
            self.gamma_s, self.v_hs, v_syn)*(_V - self.e_syn))

        #: dV
        _dV = -(I_nap + I_k + I_q + I_leak + I_syn + I_app)/self.c_m

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
