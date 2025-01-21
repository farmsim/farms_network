""" Network """

import numpy as np

from .data import NetworkData
from .network_cy import NetworkCy
from .options import NetworkOptions


class Network(NetworkCy):
    """ Network class """

    def __init__(self, network_options: NetworkOptions):
        """ Initialize """

        super().__init__(network_options)
        self.data = NetworkData.from_options(network_options)

        self.nodes: list = []
        self.edges = []
        self.nodes_output_data = []
        self.__tmp_node_outputs = np.zeros((self.c_network.nnodes,))
        self.setup_network(network_options, self.data)

        # Integration options
        self.n_iterations: int = network_options.integration.n_iterations
        self.timestep: int = network_options.integration.timestep
        self.iteration: int = 0
        self.buffer_size: int = network_options.logs.buffer_size

        # Set the seed for random number generation
        random_seed = network_options.random_seed
        # np.random.seed(random_seed)

    @classmethod
    def from_options(cls, options: NetworkOptions):
        """ Initialize network from NetworkOptions """
        return cls(options)

    def to_options(self):
        """ Return NetworkOptions from network """
        return self.options

    def setup_integrator(self, network_options: NetworkOptions):
        """ Setup integrator for neural network """
        # Setup ODE numerical integrator
        integration_options = network_options.integration
        timestep = integration_options.timestep
        self.ode_integrator = RK4Solver(self.c_network.nstates, timestep)
        # Setup SDE numerical integrator for noise models if any
        noise_options = []
        for node in network_options.nodes:
            if node.noise is not None:
                if node.noise.is_stochastic:
                    noise_options.append(node.noise)

        self.sde_system = OrnsteinUhlenbeck(noise_options)
        self.sde_integrator = EulerMaruyamaSolver(len(noise_options), timestep)

    @staticmethod
    def generate_node(node_options: NodeOptions):
        """ Generate a node from options """
        Node = NodeFactory.create(node_options.model)
        node = Node.from_options(node_options)
        return node

    @staticmethod
    def generate_edge(edge_options: EdgeOptions, nodes_options):
        """ Generate a edge from options """
        target = nodes_options[nodes_options.index(edge_options.target)]
        Edge = EdgeFactory.create(target.model)
        edge = Edge.from_options(edge_options)
        return edge
