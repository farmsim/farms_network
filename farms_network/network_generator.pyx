# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=True
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: overflowcheck=False


""" Generate neural network. """
from libc.stdio cimport printf
from farms_network_generator.neuron cimport Neuron
from farms_dae_generator.parameters cimport Parameters
from cython.parallel import prange
from farms_network_generator.leaky_integrator cimport LeakyIntegrator
import farms_pylog as pylog
from farms_network_generator.neuron_factory import NeuronFactory
from collections import OrderedDict
import numpy as np
cimport numpy as cnp
cimport cython
pylog.set_level('debug')


cdef class NetworkGenerator(object):
    """ Generate Neural Network.
    """

    def __init__(self, dae, graph):
        """Initialize.

        Parameters
        ----------
        graph_file_path: <str>
            File path to the graphml structure.
        """
        super(NetworkGenerator, self).__init__()

        #: Attributes
        self.neurons = OrderedDict()  #: Neurons in the network
        self.dae = dae

        self.x = <Parameters > self.dae.x
        self.c = <Parameters > self.dae.c
        self.u = <Parameters > self.dae.u
        self.p = <Parameters > self.dae.p
        self.y = <Parameters > self.dae.y
        self.xdot = <Parameters > self.dae.xdot

        self.odes = []

        self.fin = {}
        self.integrator = {}

        #:  Read the graph
        self.graph = graph

        #: Get the number of neurons in the model
        self.num_neurons = len(self.graph)

        self.c_neurons = np.ndarray((self.num_neurons,), dtype=Neuron)
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
            pylog.debug(
                'Generating neuron model : {} of type {}'.format(
                    name, neuron['model']))
            #: Generate Neuron Models
            _neuron = factory.gen_neuron(neuron['model'])
            self.neurons[name] = _neuron(name, self.dae,
                                         self.graph.in_degree(name),
                                         **neuron)
            self.c_neurons[j] = <Neuron > self.neurons[name]

    def generate_network(self):
        """
        Generate the network.
        """
        for name, neuron in list(self.neurons.items()):
            pylog.debug(
                'Establishing neuron {} network connections'.format(
                    name))
            for j, pred in enumerate(self.graph.predecessors(name)):
                pylog.debug(('{} -> {}'.format(pred, name)))
                #: Set the weight of the parameter
                neuron.add_ode_input(
                    j, self.neurons[pred],
                    self.dae, **self.graph[pred][name])

    def ode(self, t, cnp.ndarray[double, ndim=1] state):
        return self.c_ode(t, state)

    #################### C-FUNCTIONS ####################

    cdef double[:] c_ode(self, double t, double[:] state):
        self.x.c_set_values(state)
        cdef unsigned int j
        cdef Neuron n

        for j in range(self.num_neurons):
            n = self.c_neurons[j]
            n.c_output()

        for j in range(self.num_neurons):
            n = self.c_neurons[j]
            n.c_ode_rhs(self.y.c_get_values(),
                        self.p.c_get_values())
        return self.xdot.c_get_values()
