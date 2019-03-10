"""Implementation of integrate and fire neuron type.
Inherits from Neuron() abstract class. """

from neuron import Neuron
import numpy as np


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
