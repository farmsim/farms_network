""" Edge """

from farms_network.core.edge_cy import EdgeCy
from farms_network.core.options import EdgeOptions


class Edge(EdgeCy):
    """ Interface to edge class """

    def __init__(self, source: str, target: str, edge_type: str, **kwargs):
        super().__init__(source, target, edge_type, **kwargs)

    @classmethod
    def from_options(cls, edge_options: EdgeOptions):
        """ From edge options """
        source: str = edge_options.source
        target: str = edge_options.target
        edge_type: str = edge_options.type
        parameter_options = {} if edge_options.parameters is None else edge_options.parameters
        return cls(source, target, edge_type, **parameter_options)
