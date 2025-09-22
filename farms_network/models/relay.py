""" Relay """


from farms_network.core.options import RelayNodeOptions
from farms_network.models.relay_cy import RelayNodeCy
from farms_network.core.node import Node
from farms_network.models import Models


class RelayNode(Node):
    """ Relay node Cy """

    CY_NODE_CLASS = RelayNodeCy

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, model=Models.RELAY, **kwargs)
