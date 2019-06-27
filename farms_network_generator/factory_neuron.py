"""Factory class for generating the neuron model."""

from lif_danner import LIFDanner
from lif_danner_nap import LIFDannerNap
from lif_daun_interneuron import LIFDaunInterneuron
from hh_daun_motorneuron import HHDaunMotorneuron
from constant_and_inhibit import ConstantAndInhibit
from sensory_neuron import SensoryNeuron
from integrate_and_fire import IntegrateAndFire
from leaky_integrator import LeakyIntegrator
from oscillator import Oscillator


class FactoryNeuron(object):
    """Implementation of Factory Neuron class.
    """

    def __init__(self, neuron_type):
        super(FactoryNeuron, self).__init__()

        self.neuron_type = neuron_type
