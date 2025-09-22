from farms_network.core.node import Node
from farms_network.models import Models
from farms_network.core.options import HopfOscillatorNodeOptions
from farms_network.models.hopf_oscillator_cy import HopfOscillatorNodeCy


class HopfOscillatorNode(Node):

    CY_NODE_CLASS = HopfOscillatorNodeCy

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, model=Models.OSCILLATOR, **kwargs)

    # Hopf Oscillator-specific properties
