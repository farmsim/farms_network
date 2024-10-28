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

from farms_network.core.edge import PyEdge
# from farms_network.models.fitzhugh_nagumo import FitzhughNagumo
# from farms_network.models.hh_daun_motoneuron import HHDaunMotoneuron
# from farms_network.models.hopf_oscillator import HopfOscillator
# from farms_network.models.leaky_integrator import LeakyIntegrator
from farms_network.models.external_relay import PyExternalRelayNode
from farms_network.models.linear import PyLinearNode
from farms_network.models.li_danner import PyLIDannerNode
from farms_network.models.li_nap_danner import PyLINaPDannerNode
# from farms_network.models.lif_daun_interneuron import LIFDaunInterneuron
# from farms_network.models.matsuoka_node import MatsuokaNode
# from farms_network.models.morphed_oscillator import MorphedOscillator
# from farms_network.models.morris_lecar import MorrisLecarNode
from farms_network.models.oscillator import PyOscillatorNode, PyOscillatorEdge
# from farms_network.models.relu import ReLUNode
# from farms_network.models.sensory_node import SensoryNode


class NodeFactory:
    """Implementation of Factory Node class.
    """
    nodes = {
        # 'if': IntegrateAndFire,
        'oscillator': PyOscillatorNode,
        # 'hopf_oscillator': HopfOscillator,
        # 'morphed_oscillator': MorphedOscillator,
        # 'leaky': LeakyIntegrator,
        'external_relay': PyExternalRelayNode,
        'linear': PyLinearNode,
        'li_nap_danner': PyLINaPDannerNode,
        'li_danner': PyLIDannerNode,
        # 'lif_daun_interneuron': LIFDaunInterneuron,
        # 'hh_daun_motoneuron': HHDaunMotoneuron,
        # 'fitzhugh_nagumo': FitzhughNagumo,
        # 'matsuoka_node': MatsuokaNode,
        # 'morris_lecar': MorrisLecarNode,
        # 'relu': ReLUNode,
    }

    def __init__(self):
        """Factory initialization."""
        super(NodeFactory, self).__init__()

    @staticmethod
    def register_node(node_type, node_instance):
        """
        Register a new type of node that is a child class of Node.
        Parameters
        ----------
        self: type
            description
        node_type: <str>
            String to identifier for the node.
        node_instance: <cls>
            Class of the node to register.
        """
        NodeFactory.nodes[node_type] = node_instance

    @staticmethod
    def generate_node(node_type):
        """Generate the necessary type of node.
        Parameters
        ----------
        self: type
            description
        node_type: <str>
            One of the following list of available nodes.
            1. if - Integrate and Fire
            2. lif_danner_nap - LIF Danner Nap
            3. lif_danner - LIF Danner
            4. lif_daun_internode - LIF Daun Internode
            5. hh_daun_motornode - HH_Daun_Motornode
        Returns
        -------
        node: <cls>
            Appropriate node class.
        """
        node = NodeFactory.nodes.get(node_type)
        if not node:
            raise ValueError(node_type)
        return node


class EdgeFactory:
    """Implementation of Factory Edge class.
    """
    edges = {
        'oscillator': PyOscillatorEdge,
    }

    def __init__(self):
        """Factory initialization."""
        super().__init__()

    @staticmethod
    def register_edge(edge_type, edge_instance):
        """
        Register a new type of edge that is a child class of Edge.
        Parameters
        ----------
        self: type
            description
        edge_type: <str>
            String to identifier for the edge.
        edge_instance: <cls>
            Class of the edge to register.
        """
        EdgeFactory.edges[edge_type] = edge_instance

    @staticmethod
    def generate_edge(edge_type):
        """Generate the necessary type of edge.
        Parameters
        ----------
        edge_type: <str>
        Returns
        -------
        edge: <cls>
            Appropriate edge class.
        """
        edge = EdgeFactory.edges.get(edge_type, PyEdge)
        if not edge:
            raise ValueError(edge_type)
        return edge
