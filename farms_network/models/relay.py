""" Relay """


from farms_network.core.options import RelayNodeOptions
from farms_network.models.relay_cy import RelayNodeCy


class RelayNode:
    """ Relay node Cy """

    def __init__(self, name: str, **kwargs):
        self._node_cy: RelayNodeCy = RelayNodeCy(name, **kwargs)

    @classmethod
    def from_options(cls, node_options: RelayNodeOptions):
        """ Instantiate relay node from options """
        name: str = node_options.name
        return cls(name)
