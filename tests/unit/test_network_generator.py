"""Unit test of for generic neuron model."""

import unittest
import ddt
from farms_container import Container
from farms_network.neuron_factory import NeuronFactory
from farms_network.network_generator import NetworkGenerator
import numpy as np
import farms_pylog as pylog
import networkx as nx

# pylog.set_level('debug')

def neuron_types():
    """ Return all the muscle types implemented. """
    return (list(NeuronFactory.neurons.keys()))

@ddt.ddt
class TestNetworkGenerator(unittest.TestCase):
    """Test different neuron models"""

    def setUp(self):
        """ Set up the neruon model. """
        self.container = Container()
        self.container.add_namespace('neural')

        #: Create a network graph
        self.network = nx.DiGraph()

    def tearDown(self):
        """ Tear up neuron model. """
        pylog.debug("Tear down...")
        Container.del_instance()

    @ddt.data(*neuron_types())
    def test_generate_neurons(self, value):
        """ Test neuron generation. """

        #: Add two neurons to graph        
        self.network.add_node('neuron_1', model=value)
        self.network.add_node('neuron_2', model=value)
        #: Connect two neurons with default weights
        self.network.add_edge('neuron_1', 'neuron_2')
        self.network.add_edge('neuron_2', 'neuron_1')
        #: Generate network
        generator  = NetworkGenerator(self.network)
        # self.container.initialize()
        self.assertIsNone(generator.generate_neurons())

    @ddt.data(*neuron_types())
    def test_generate_network(self, value):
        """ Test network creation. """
        #: Add two neurons to graph        
        self.network.add_node('neuron_1', model=value)
        self.network.add_node('neuron_2', model=value)
        #: Connect two neurons with default weights
        self.network.add_edge('neuron_1', 'neuron_2')
        self.network.add_edge('neuron_2', 'neuron_1')
        #: Generate network
        generator  = NetworkGenerator(self.network)
        # self.container.initialize()
        self.assertIsNone(generator.generate_network())

    @ddt.data(*neuron_types())
    def test_ode(self, value):
        """ Test network ode. """
        #: Add two neurons to graph        
        self.network.add_node('neuron_1', model=value)
        self.network.add_node('neuron_2', model=value)
        #: Connect two neurons with default weights
        self.network.add_edge('neuron_1', 'neuron_2')
        self.network.add_edge('neuron_2', 'neuron_1')
        #: Generate network
        generator  = NetworkGenerator(self.network)
        self.container.initialize()
        self.assertIsNotNone(
            generator.ode(0.0,
                          np.array(self.container.neural.states.values)))

if __name__ == '__main__':        
    unittest.main()
