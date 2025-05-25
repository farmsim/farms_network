from farms_network.core.node import Node
from farms_network.models import Models
from farms_network.core.options import ReLUNodeOptions
from farms_network.models.relu_cy import ReLUNodeCy


class ReLUNode(Node):

    CY_NODE_CLASS = ReLUNodeCy

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, model=Models.RELU, **kwargs)

    # ReLU-specific properties
    @property
    def gain(self):
        return self._node_cy.gain

    @gain.setter
    def gain(self, value):
        self._node_cy.gain = value

    @property
    def sign(self):
        return self._node_cy.sign

    @sign.setter
    def sign(self, value):
        self._node_cy.sign = value

    @property
    def offset(self):
        return self._node_cy.offset

    @offset.setter
    def offset(self, value):
        self._node_cy.offset = value
