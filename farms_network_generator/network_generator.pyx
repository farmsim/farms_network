""" Generate neural network. """
# cython: profile=True

from farms_network_generator.leaky_integrator cimport LeakyIntegrator
import itertools
from scipy.integrate import ode
import farms_pylog as biolog
from farms_network_generator.neuron_factory import NeuronFactory
from collections import OrderedDict
import numpy as np
from farms_dae_generator.dae_generator import DaeGenerator
from farms_dae_generator.parameters cimport Parameters
from networkx_model import NetworkXModel
cimport cython
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
        self.c_neurons = np.ndarray((4,), dtype=LeakyIntegrator)
        self.dae = DaeGenerator()

        self.odes = []

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

        for j, (name, neuron) in enumerate(sorted(self.graph.node.items())):
            #: Add neuron to list

            print(name, neuron.keys())
            biolog.debug(
                'Generating neuron model : {} of type {}'.format(
                    name, neuron['model']))
            #: Generate Neuron Models
            _neuron = factory.gen_neuron(neuron['model'])
            self.neurons[name] = _neuron(name, self.dae, **neuron)
            self.c_neurons[j] = <LeakyIntegrator > self.neurons[name]

    def generate_network(self):
        """
        Generate the network.
        """
        for name, neuron in list(self.neurons.items()):
            biolog.debug(
                'Establishing neuron {} network connections'.format(
                    name))
            for j, pred in enumerate(self.graph.predecessors(name)):
                print(('{} -> {}'.format(pred, name)))
                neuron.add_ode_input(
                    self.dae, self.neurons[pred], j, **self.graph[pred][name])

    def setup_integrator(self, x0, integrator='dopri853', atol=1e-20,
                         rtol=1e-20):
        """Setup system."""
        self.dae.initialize_dae()
        self.integrator = ode(self.ode).set_integrator(
            integrator,
            # method='bdf',
            atol=atol,
            rtol=rtol)
        self.integrator.set_initial_value(x0, 0.0)

        # for neuron in self.neurons.values():
        #     self.odes.append(*neuron.ode_rhs)
        #     print(*neuron.ode_rhs)

    @cython.profile(True)
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    @cython.cdivision(False)
    cdef void c_step(self):
        """Step ode system. """
        cdef unsigned int j
        for j in range(4):
            self.c_neurons[j].ode_rhs()

    @cython.profile(True)
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    @cython.cdivision(False)
    cdef real[:] c_ode(self, real t, real[:] state):
        cdef Parameters x = <Parameters > self.dae.x
        cdef Parameters y = <Parameters > self.dae.y
        x.c_set_values(state)
        self.c_step()
        return y.c_get_values()

    def ode(self, t, state):
        return self.c_ode(t, state)

    def step(self):
        """Step integrator."""
        time = self.integrator.t
        dt = 0.001
        self.integrator.integrate(time+dt)
        assert(self.integrator.successful())
        self.dae.update_log()
