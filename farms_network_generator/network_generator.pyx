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

        self.x = <Parameters > self.dae.x
        self.c = <Parameters > self.dae.c
        self.u = <Parameters > self.dae.u
        self.p = <Parameters > self.dae.p
        self.y = <Parameters > self.dae.y
        self.xdot = <Parameters > self.dae.xdot

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

    def setup_integrator(self, x0, integrator='dopri853', atol=1e-6,
                         rtol=1e-6, method='adams'):
        """Setup system."""
        self.dae.initialize_dae()
        self.integrator = ode(self.ode).set_integrator(
            integrator,
            method=method,
            atol=atol,
            rtol=rtol)
        self.integrator.set_initial_value(x0, 0.0)

    @cython.profile(True)
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    @cython.cdivision(False)
    cdef void c_step(self, real[:] inputs):
        """Step ode system. """
        cdef double time = self.integrator.t
        cdef double dt = 0.001
        self.dae.u.values = inputs
        self.dae.x.c_set_values(self.integrator.integrate(time+dt))
        self.integrator.set_initial_value(self.integrator.y,
                                          self.integrator.t)
        assert(self.integrator.successful())
        self.x.c_update_log()
        self.p.c_update_log()
        self.c.c_update_log()
        self.u.c_update_log()
        self.y.c_update_log()

    @cython.profile(True)
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    @cython.cdivision(False)
    cdef c_ode(self, real t, real[:] state):
        self.dae.x.values = state
        cdef unsigned int j
        for j in range(4):
            self.c_neurons[j].c_ode_rhs(self.y, self.p)

    @cython.profile(True)
    def ode(self, t, state):
        self.c_ode(t, state)
        return self.dae.y.values

    @cython.profile(True)
    def step(self, u):
        """Step integrator."""
        self.c_step(u)
