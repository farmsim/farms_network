""" Edge """

from farms_network.core.edge_cy import EdgeCy
from farms_network.core.options import EdgeOptions
from farms_network.models import EdgeTypes
from typing import Dict, Type


class Edge:
    """ Interface to edge class """

    CY_EDGE_CLASS: Type[EdgeCy] = None

    def __init__(self, source: str, target: str, edge_type: EdgeTypes, model, **kwargs):
        self.model = model
        self.source: str = source
        self.target: str = target
        self._edge_cy = self._create_cy_edge(edge_type, **kwargs)

    def _create_cy_edge(self, edge_type, **kwargs) -> EdgeCy:
        if self.CY_EDGE_CLASS is None:
            return EdgeCy(edge_type, **kwargs)
        return self.CY_EDGE_CLASS(edge_type, **kwargs)

    @property
    def edge_type(self):
        return self._edge_cy.type

    @edge_type.setter
    def edge_type(self, edge_type: EdgeTypes):
        self._edge_cy.type = edge_type

    @classmethod
    def from_options(cls, edge_options: EdgeOptions):
        """ From edge options """
        model = edge_options.model
        source: str = edge_options.source
        target: str = edge_options.target
        edge_type: EdgeTypes = edge_options.type
        # Need to generate parameters based on the model specified
        parameter_options: Dict = {} if edge_options.parameters is None else edge_options.parameters
        return cls(source, target, edge_type, model, **parameter_options)
