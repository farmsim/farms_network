""" Class to Generate network and integrate over time. """

from farms_network.network_generator import NetworkGenerator
from scipy.integrate import ode
from farms_container import Container
from .networkx_model import NetworkXModel


class NeuralSystem(NetworkXModel):
    """Neural System.
    """

    def __init__(self, config_path):
        """ Initialize neural system. """
        super(NeuralSystem, self).__init__()
        self.container = Container.get_instance()
        #: Add name-space for neural system data
        self.container.add_namespace('neural')
        self.config_path = config_path
        self.integrator = None
        self.read_graph(config_path)
        self.network = NetworkGenerator(self.graph)

    def setup_integrator(self, x0=None, integrator='dopri5', atol=1e-6,
                         rtol=1e-6, method='adams'):
        """Setup system."""
        self.integrator = ode(self.network.ode).set_integrator(
            integrator,
            method=method,
            atol=atol,
            rtol=rtol)

        if not x0:
            self.integrator.set_initial_value(
                self.container.neural.states.values, 0.0)
        else:
            self.integrator.set_initial_value(x0, 0.0)

    def step(self, dt=1, update=True):
        """Step ode system. """
        self.integrator.set_initial_value(self.integrator.y,
                                          self.integrator.t)
        self.integrator.integrate(self.integrator.t+dt)
        #: Update the logs
        if update:
            #: TO-DO
            self.container.neural.update_log()
