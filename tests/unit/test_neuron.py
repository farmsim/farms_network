"""Unit test of for generic neuron model."""

import unittest
import ddt
from farms_network.neuron_factory import NeuronFactory
from farms_container import Container
import numpy as np
import farms_pylog as pylog

def neuron_types():
    """ Return all the muscle types implemented. """
    return list(NeuronFactory.neurons.keys())

@ddt.ddt
class TestMuscleModel(unittest.TestCase):
    """Test different muscle models in the framework.
    """

    def setUp(self):
        """ Set up the muscle model. """
        pass

    @ddt.data(*neuron_types())
    def test_neuron_instancce(self, value):
        """ Test neurons names. """
        pylog.debug("Initializing neurons {}".format(value))
        Neuron = NeuronFactory.gen_neuron(value)
        neuron = Neuron("test_neuron", 0)
        self.assertIsInstance(neuron, Neuron)

if __name__ == '__main__':    
    container = Container.get_instance()
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
    unittest.main()
