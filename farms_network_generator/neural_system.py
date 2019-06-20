""" Class to Generate network and integrate over time. """

from farms_network_generator.network_generator import NetworkGenerator
from scipy.integrate import ode


class NeuralSystem(object):
    """Neural System.
    """

    def __init__(self, dae, config_path):
        """ Initialize neural system. """
        super(NeuralSystem, self).__init__()
        self.dae = dae
        self.config_path = config_path
        self.network = NetworkGenerator(dae, config_path)

    def setup_integrator(self, x0, integrator='dopri5', atol=1e-6,
                         rtol=1e-6, method='adams'):
        """Setup system."""
        self.integrator = ode(self.network.ode).set_integrator(
            integrator,
            method=method,
            atol=atol,
            rtol=rtol)
        self.integrator.set_initial_value(x0, 0.0)

    def step(self, dt=1):
        """Step ode system. """
        self.integrator.set_initial_value(self.integrator.y,
                                          self.integrator.t)
        self.integrator.integrate(self.integrator.t+dt)
