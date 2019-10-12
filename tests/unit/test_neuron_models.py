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
class TestNeuronModels(unittest.TestCase):
    """Test different neuron models"""

    def setUp(self):
        """ Set up the neruon model. """
        container = Container()
        container.add_namespace('neural')
        #: ODE States
        container.neural.add_table('states')
        container.neural.add_table('dstates')
        #: Weights
        container.neural.add_table('weights')
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
    def test_neuron_output_method(self, value):
        """ Test if neuron output method exists . """
        pylog.debug("Initializing neurons {}".format(value))
        Neuron = NeuronFactory.gen_neuron(value)
        neuron = Neuron("test_neuron", 0)
        container = Container.get_instance()
        container.initialize()
        self.assertIsNone(neuron.output())

    @ddt.data(*neuron_types())
    def test_neuron_ode_rhs_method(self, value):
        """ Test if neuron has ode_rhs method implemented """
        pylog.debug("Initializing neurons {}".format(value))
        Neuron = NeuronFactory.gen_neuron(value)
        neuron = Neuron("test_neuron", 0)
        container = Container.get_instance()
        container.initialize()
        self.assertIsNone(neuron.ode_rhs(container.neural.outputs.values,
                                         container.neural.parameters.values))

    @ddt.data(*neuron_types())
    def test_neuron_add_ode_input_method(self, value):
        """ Test neurons add_ode_input. """
        pylog.debug("Initializing neurons {}".format(value))
        Neuron = NeuronFactory.gen_neuron(value)
        neuron = Neuron("test_neuron", 1)
        container = Container.get_instance()
        container.initialize()
        self.assertIsNone(neuron.add_ode_input(1, neuron))

    @ddt.data(*neuron_types())
    def test_neuron_model_type(self, value):
        """ Test neuron model type. """
        pylog.debug("Initializing neurons {}".format(value))
        Neuron = NeuronFactory.gen_neuron(value)
        neuron = Neuron("test_neuron", 0)
        container = Container.get_instance()
        container.initialize()
        self.assertIn(neuron.model_type, neuron_types())

if __name__ == '__main__':        
    unittest.main()
