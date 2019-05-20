"""Constant and Inhibit neuron type."""

from neuron import Neuron


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
