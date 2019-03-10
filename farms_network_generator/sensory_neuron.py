"""Sensory afferent neurons."""

from neuron import Neuron


class SensoryNeuron(Neuron):
    """Sensory afferent neurons connecting muscle model with the network.
    """

    def __init__(self, n_id, dae, **kwargs):
        """Initialize.
        Parameters
        ----------
        n_id: str
            Unique ID for the neuron in the network.
        """
        super(SensoryNeuron, self).__init__(
            neuron_type='sensory_neuron')

        #: Neuron ID
        self.n_id = n_id
        self.dae = dae

        #: Initialize parameters
        self.weight = self.dae.add_c('weight_' + self.n_id,
                                     kwargs.get('weight', 0.1),
                                     param_type='val')

        self.aff_inp = self.dae.add_p('aff_' + self.n_id,
                                      kwargs.get('init', 0.0),
                                      param_type='sym')

    def add_ode_input(self, neuron, **kwargs):
        """Abstract method"""
        pass

    def ode_rhs(self):
        """Abstract method"""
        pass

    def ode_alg_eqn(self):
        """Abstract method"""
        pass

    def neuron_out(self):
        """ Output of the neuron model.
        Returns
        ----------
        out: <cas.SX.sym>
            Output of the neuron  model
        """
        return self.aff_inp.sym
