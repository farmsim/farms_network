"""Leaky Integrate and Fire Motorneuron."""

from neuron import Neuron
import casadi as cas
import farms_pylog as biolog


class HHDaunMotorneuron(Neuron):
    """Leaky Integrate and Fire Interneuron.
    Based on Silvia Daun and Tbor's model.
    """

    def __init__(self, n_id, dae, **kwargs):
        super(HHDaunMotorneuron, self).__init__(
            neuron_type='hh_daun_motorneuron')

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
