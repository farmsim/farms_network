"""Unit test of for generic neuron model."""

import unittest
import farms_pylog as pylog
from farms_network.neuron import Neuron
# pylog.set_level('debug')

class TestNeuronBaseClass(unittest.TestCase):
    """Test neuron base class
    """

    def test_neuron_instance(self):
        """ Test neurons instances. """
        neuron = Neuron("test_neuron")
        self.assertIsInstance(neuron, Neuron)

    def test_neuron_output_abstract_method(self):
        """ Test if neuron output method exists . """
        neuron = Neuron("test_neuron")
        with self.assertRaises(NotImplementedError):
            neuron.output()

    def test_neuron_ode_rhs_abstract_method(self):
        """ Test if neuron has ode_rhs method implemented """
        neuron = Neuron("test_neuron")
        with self.assertRaises(NotImplementedError):
            neuron.ode_rhs(None, None)

    def test_neuron_add_ode_input_exists(self):
        """ Test neurons add_ode_input. """
        neuron = Neuron("test_neuron")
        with self.assertRaises(NotImplementedError):
            neuron.add_ode_input(None)

    def test_neuron_model_type(self):
        """ Test neuron model type. """
        neuron = Neuron("test_neuron")
        self.assertIn(neuron.model_type, "test_neuron")

if __name__ == '__main__':
    unittest.main()
