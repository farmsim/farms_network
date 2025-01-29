""" Node """


from farms_network.core.node_cy import NodeCy
from farms_network.core.options import NodeOptions, NodeParameterOptions


class Node(NodeCy):

    def __init__(self, name: str):
        super().__init__(name)

    @classmethod
    def from_options(cls, node_options: NodeOptions):
        """ From node options """
        name: str = node_options.name
        parameters: NodeParameterOptions = node_options.parameters if node_options.parameters else {}
        return cls(name, **parameters)
