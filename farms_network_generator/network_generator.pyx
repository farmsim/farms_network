""" Generate neural network. """
# cython: profile=True

import itertools
from scipy.integrate import ode
import farms_pylog as biolog
from farms_network_generator.neuron_factory import NeuronFactory
from collections import OrderedDict
import numpy as np
from farms_dae_generator.dae_generator import DaeGenerator
from networkx_model import NetworkXModel

biolog.set_level('debug')


cdef class NetworkGenerator(object):
    """ Generate Neural Network.
    """

    def __init__(self, graph_file_path):
        """Initialize.

        Parameters
        ----------
        graph_file_path: <str>
            File path to the graphml structure.
        """
        super(NetworkGenerator, self).__init__()

        self.nx = NetworkXModel()

        #: Attributes
        self.neurons = OrderedDict()  #: Neurons in the network
        self.dae = DaeGenerator()

        self.fin = {}
        self.integrator = {}

        #: METHODS
        self.graph = self.nx.read_graph(graph_file_path)
        print(graph_file_path)
        self.generate_neurons()
        self.generate_network()

    def generate_neurons(self):
        """Generate the complete neural network.
        Instatiate a neuron model for each node in the graph

        Returns
        -------
        out : <bool>
            Return true if successfully created the neurons
        """

        factory = NeuronFactory()

        for name, neuron in sorted(self.graph.node.items()):
            #: Add neuron to list

            print(name, neuron.keys())
            biolog.debug(
                'Generating neuron model : {} of type {}'.format(
                    name, neuron['model']))
            #: Generate Neuron Models
            _neuron = factory.gen_neuron(neuron['model'])
            self.neurons[name] = _neuron(name, self.dae, **neuron)

    def generate_network(self):
        """
        Generate the network.
        """
        for name, neuron in list(self.neurons.items()):
            biolog.debug(
                'Establishing neuron {} network connections'.format(
                    name))
            for pred in self.graph.predecessors(name):
                print(('{} -> {}'.format(pred, name)))
                neuron.add_ode_input(
                    self.neurons[pred], **self.graph[pred][name])

    def setup_integrator(self, x0):
        """Setup system."""
        self.dae.initialize_dae()
        self.integrator = ode(self.ode).set_integrator(
            'dopri5',
            # method='bdf',
            atol=1e-6,
            rtol=1e-6)
        self.integrator.set_initial_value(x0, 0.0)

    cdef void c_step(self):
        """Step ode system. """
        cdef LeakyIntegrator neuron
        cdef unsigned int j
        for j, neuron in enumerate(self.neurons.values()):
            neuron.ode_rhs()

    def ode(self, t, state):
        self.dae.x.values = np.array(state, dtype=np.float)
        self.c_step()
        return self.dae.y.values

    def step(self):
        """Step integrator."""
        time = self.integrator.t
        dt = 0.001
        self.integrator.integrate(time+dt)
        self.dae.update_log()
