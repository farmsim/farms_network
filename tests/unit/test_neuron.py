"""Unit test of for generic neuron model."""

import unittest
import ddt
from farms_network.neuron_factory import NeuronFactory
from farms_container import Container
import numpy as np
import farms_pylog as pylog

# pylog.set_level('debug')

def neuron_types():
    """ Return all the muscle types implemented. """
    return (list(NeuronFactory.neurons.keys()))

@ddt.ddt
class TestNeuronModel(unittest.TestCase):
    """Test different neuron models in the framework.
    """

    def setUp(self):
        """ Set up the neruon model. """
        container = Container()
        container.add_namespace('neural')
        #: ODE States
        container.neural.add_table('states')
        container.neural.add_table('dstates')
        #: Neurons parameters
        container.neural.add_table(
            'constants', TABLE_TYPE='CONSTANT')
        container.neural.add_table('parameters')
        #: Input to neural system
        container.neural.add_table('inputs')
        #: Secondary outputs 
        container.neural.add_table('outputs')        

    def tearDown(self):
        """ Tear up neuron model. """
        pylog.debug("Tear down...")
        Container.del_instance()

    @ddt.data(*neuron_types())
    def test_neuron_instance(self, value):
        """ Test neurons instances. """
        pylog.debug("Initializing neurons {}".format(value))
        Neuron = NeuronFactory.gen_neuron(value)
        neuron = Neuron("test_neuron", 0)
        container = Container.get_instance()
        container.initialize()
        self.assertIsInstance(neuron, Neuron)

    @ddt.data(*neuron_types())
    def test_neuron_output(self, value):
        """ Test neurons output. """
        pylog.debug("Initializing neurons {}".format(value))
        Neuron = NeuronFactory.gen_neuron(value)
        neuron = Neuron("test_neuron", 0)
        container = Container.get_instance()
        container.initialize()
        self.assertIsNone(neuron.output())

    @ddt.data(*neuron_types())
    def test_neuron_ode_rhs(self, value):
        """ Test neurons ode_rhs. """
        pylog.debug("Initializing neurons {}".format(value))
        Neuron = NeuronFactory.gen_neuron(value)
        neuron = Neuron("test_neuron", 0)
        container = Container.get_instance()
        container.initialize()
        self.assertIsNone(neuron.ode_rhs(container.neural.outputs.values,
                                         container.neural.parameters.values))

    @ddt.data(*neuron_types())
    def test_neuron_add_ode_input(self, value):
        """ Test neurons add_ode_input. """
        pylog.debug("Initializing neurons {}".format(value))
        Neuron = NeuronFactory.gen_neuron(value)
        neuron = Neuron("test_neuron", 0)
        container = Container.get_instance()
        container.initialize()
        self.assertIsNotNone(neuron.add_ode_input)

if __name__ == '__main__':        
    unittest.main()
