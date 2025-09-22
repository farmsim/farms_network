from farms_network.core.node import Node
from farms_network.core.edge import Edge
from farms_network.models import Models
from farms_network.core.options import OscillatorNodeOptions
from farms_network.models.oscillator_cy import OscillatorNodeCy
from farms_network.models.oscillator_cy import OscillatorEdgeCy


class OscillatorNode(Node):

    CY_NODE_CLASS = OscillatorNodeCy

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, model=Models.OSCILLATOR, **kwargs)

    # Oscillator-specific properties


class OscillatorEdge(Edge):

    CY_EDGE_CLASS = OscillatorEdgeCy

    def __init__(self, source, target, edge_type, model=Models.OSCILLATOR, **kwargs):
        super().__init__(
            source=source, target=target, edge_type=edge_type, model=Models.OSCILLATOR, **kwargs
        )
