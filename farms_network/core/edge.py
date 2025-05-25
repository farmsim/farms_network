""" Edge """

from farms_network.core.edge_cy import EdgeCy
from farms_network.core.options import EdgeOptions
from farms_network.models import EdgeTypes
from typing import Dict


class Edge(EdgeCy):
    """ Interface to edge class """

    def __init__(self, source: str, target: str, edge_type: EdgeTypes, **kwargs):
        self.source: str = source
        self.target: str = target
        self._edge_cy = EdgeCy(source, target, edge_type)

    @property
    def edge_type(self):
        return self._edge_cy.type

    @edge_type.setter
    def edge_type(self, edge_type: EdgeTypes):
        self._edge_cy.type = edge_type

    @classmethod
    def from_options(cls, edge_options: EdgeOptions):
        """ From edge options """
        source: str = edge_options.source
        target: str = edge_options.target
        edge_type: EdgeTypes = edge_options.type
        # Need to generate parameters based on the model specified
        parameter_options: Dict = {} # if edge_options.parameters is None else edge_options.parameters
        return cls(source, target, edge_type, **parameter_options)
