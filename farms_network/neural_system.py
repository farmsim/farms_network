import numpy as np
from farms_network.network_generator import NetworkGenerator
from scipy.integrate import ode
from .networkx_model import NetworkXModel


class NeuralSystem(NetworkXModel):
    """Neural System.
    """

    def __init__(self, config_path, container):
        """ Initialize neural system. """
        super(NeuralSystem, self).__init__()
        self.container = container
        #: Add name-space for neural system data
        neural_table = self.container.add_namespace('neural')
        self.config_path = config_path
        self.integrator = None
        self.read_graph(config_path)
        #: Create network
        self.network = NetworkGenerator(self.graph, neural_table)

    def setup_integrator(
            self, x0=None, integrator=u'dopri5', atol=1e-12, rtol=1e-6,
            max_step=0.0, method=u'adams'
    ):
        """Setup system."""
        self.integrator = ode(self.network.ode).set_integrator(
            integrator,
            method=method,
            atol=atol,
            rtol=rtol,
            max_step=max_step
        )

        if not x0:
            initial_values = np.random.rand(
                len(self.container.neural.states.values)
            )*atol
            self.integrator.set_initial_value(initial_values, 0.0)
        else:
            self.integrator.set_initial_value(x0, 0.0)

    def step(self, dt=1, update=True):
        """Step ode system. """
        self.integrator.set_initial_value(self.integrator.y,
                                          self.integrator.t)
        self.integrator.integrate(self.integrator.t+dt)
