""" Class to Generate network and integrate over time. """

from farms_network_generator.network_generator import NetworkGenerator
from scipy.integrate import ode
from farms_dae_generator.dae_generator import DaeGenerator


class NeuralSystem(object):
    """Neural System.
    """

    def __init__(self, config_path):
        """ Initialize neural system. """
        super(NeuralSystem, self).__init__()
        self.dae = DaeGenerator()
        self.config_path = config_path
        self.integrator = None
        self.network = NetworkGenerator(self.dae, config_path)

    def setup_integrator(self, x0=None, integrator='dopri5', atol=1e-6,
                         rtol=1e-6, method='adams'):
        """Setup system."""
        self.integrator = ode(self.network.ode).set_integrator(
            integrator,
            method=method,
            atol=atol,
            rtol=rtol)

        #: Initialize ode
        self.dae.initialize_dae()

        if not x0:
            self.integrator.set_initial_value(self.dae.x.values, 0.0)
        else:
            self.integrator.set_initial_value(x0, 0.0)

    def step(self, dt=1, update=True):
        """Step ode system. """
        self.integrator.set_initial_value(self.integrator.y,
                                          self.integrator.t)
        self.integrator.integrate(self.integrator.t+dt)

        #: Update the logs
        if update:
            self.dae.update_log()
