"""
-----------------------------------------------------------------------
Copyright 2018-2020 Jonathan Arreguit, Shravan Tata Ramalingasetty
Copyright 2018 BioRobotics Laboratory, École polytechnique fédérale de Lausanne

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-----------------------------------------------------------------------

Factory class for generating the node model.
"""

from typing import Dict, Optional, Type

from farms_network.core.edge import Edge
from farms_network.core.node import Node
from farms_network.models.external_relay import ExternalRelayNode
# from farms_network.models.fitzhugh_nagumo import FitzhughNagumo
# from farms_network.models.hh_daun_motoneuron import HHDaunMotoneuron
from farms_network.models.hopf_oscillator import HopfOscillatorNode
from farms_network.models.leaky_integrator import LeakyIntegratorNode
from farms_network.models.li_danner import LIDannerNode
from farms_network.models.li_nap_danner import LINaPDannerNode
from farms_network.models.linear import LinearNode
# from farms_network.models.lif_daun_interneuron import LIFDaunInterneuron
# from farms_network.models.matsuoka_node import MatsuokaNode
# from farms_network.models.morphed_oscillator import MorphedOscillator
# from farms_network.models.morris_lecar import MorrisLecarNode
from farms_network.models.oscillator import OscillatorEdge, OscillatorNode
from farms_network.models.relu import ReLUNode


class NodeFactory:
    """Implementation of Factory Node class.
    """
    _nodes: Dict[str, Type[Edge]] = {
        # 'if': PyIntegrateAndFire,
        'oscillator': OscillatorNode,
        'hopf_oscillator': HopfOscillatorNode,
        # 'morphed_oscillator': MorphedOscillator,
        'leaky_integrator': LeakyIntegratorNode,
        'external_relay': ExternalRelayNode,
        'linear': LinearNode,
        'li_nap_danner': LINaPDannerNode,
        'li_danner': LIDannerNode,
        # 'lif_daun_interneuron': LIFDaunInterneuron,
        # 'hh_daun_motoneuron': HHDaunMotoneuron,
        # 'fitzhugh_nagumo': FitzhughNagumo,
        # 'matsuoka_node': MatsuokaNode,
        # 'morris_lecar': MorrisLecarNode,
        'relu': ReLUNode,
    }

    @classmethod
    def available_types(cls) -> list[str]:
        """Get list of registered node types.

        Returns:
            Sorted list of registered node type identifiers
        """
        return sorted(cls._nodes.keys())

    @classmethod
    def create(cls, node_type: str) -> Node:
        """Create a node instance of the specified type.

        Args:
            node_type: Type identifier of node to create

        Returns:
            Instance of requested node class

        Raises:
            KeyError: If node_type is not registered
        """
        try:
            node_class = cls._nodes[node_type]
            return node_class
        except KeyError:
            available = ', '.join(sorted(cls._nodes.keys()))
            raise KeyError(
                f"Unknown node type: {node_type}. "
                f"Available types: {available}"
            )

    @classmethod
    def register(cls, node_type: str, node_class: Type[Node]) -> None:
        """Register a new node type.

        Args:
            node_type: Unique identifier for the node
            node_class: Node class to register, must inherit from Node

        Raises:
            TypeError: If node_class doesn't inherit from Node
            ValueError: If node_type is already registered
        """
        if not issubclass(node_class, Node):
            raise TypeError(f"Node class must inherit from Node: {node_class}")

        if node_type in cls._nodes:
            raise ValueError(f"Node type already registered: {node_type}")

        cls._nodes[node_type] = node_class


class EdgeFactory:
    """Implementation of Factory Edge class.
    """
    _edges: Dict[str, Type[Edge]] = {
        'oscillator': OscillatorEdge,
    }

    @classmethod
    def available_types(cls) -> list[str]:
        """Get list of registered edge types."""
        return sorted(cls._edges.keys())

    @classmethod
    def create(cls, edge_type: str) -> Edge:
        """Create an edge instance of the specified type.

        Args:
            edge_type: Type identifier of edge to create

        Returns:
            Instance of requested edge class

        Raises:
            KeyError: If edge_type is not registered
        """
        try:
            edge_class = cls._edges.get(edge_type, Edge)
            return edge_class
        except KeyError:
            available = ', '.join(sorted(cls._edges.keys()))
            raise KeyError(
                f"Unknown edge type: {edge_type}. "
                f"Available types: {available}"
            )

    @classmethod
    def register(cls, edge_type: str, edge_class: Type[Edge]) -> None:
        """Register a new edge type.

        Args:
            edge_type: Unique identifier for the edge
            edge_class: Edge class to register, must inherit from Edge

        Raises:
            TypeError: If edge_class doesn't inherit from Edge
            ValueError: If edge_type is already registered
        """
        if not issubclass(edge_class, Edge):
            raise TypeError(f"Edge class must inherit from Edge: {edge_class}")

        if edge_type in cls._edges:
            raise ValueError(f"Edge type already registered: {edge_type}")

        cls._edges[edge_type] = edge_class
