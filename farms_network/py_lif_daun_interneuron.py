"""Leaky Integrate and Fire Interneuron."""

from neuron import Neuron
import casadi as cas
import farms_pylog as biolog


class LIFDaunInterneuron(Neuron):
    """Leaky Integrate and Fire Interneuron.
    Based on Silvia Daun and Tbor's model.
    """

    def __init__(self, n_id, dae, **kwargs):
        super(LIFDaunInterneuron, self).__init__(
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
