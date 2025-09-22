""" Linear """


from farms_network.core.options import LinearNodeOptions
from farms_network.models.linear_cy import LinearNodeCy
from farms_network.core.node import Node
from farms_network.models import Models


class LinearNode(Node):
    """ Linear node Cy """

    CY_NODE_CLASS = LinearNodeCy

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, model=Models.LINEAR, **kwargs)
