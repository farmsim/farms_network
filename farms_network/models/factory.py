""" Factory class for generating the node and edges. """

from abc import ABC
from typing import Dict, Type, Union

from farms_network.core.node import Node
from farms_network.core.edge import Edge
from farms_network.models import Models
# from farms_network.models.fitzhugh_nagumo import FitzhughNagumo
# from farms_network.models.hh_daun_motoneuron import HHDaunMotoneuron
from farms_network.models.hopf_oscillator import HopfOscillatorNode
# from farms_network.models.leaky_integrator import LeakyIntegratorNode
from farms_network.models.li_danner import LIDannerNode
from farms_network.models.li_nap_danner import LINaPDannerNode
from farms_network.models.linear import LinearNode
# from farms_network.models.lif_daun_interneuron import LIFDaunInterneuron
# from farms_network.models.matsuoka_node import MatsuokaNode
# from farms_network.models.morphed_oscillator import MorphedOscillator
# from farms_network.models.morris_lecar import MorrisLecarNode
from farms_network.models.oscillator import OscillatorNode
from farms_network.models.oscillator import OscillatorEdge
from farms_network.models.relay import RelayNode
from farms_network.models.relu import ReLUNode


class BaseFactory(ABC):
    """ Base Factory implementation """

    _registry: Dict = {}

    @classmethod
    def available_types(cls) -> list[str]:
        """Get list of registered node types.

        Returns:
            Sorted list of registered node type identifiers
        """
        return list(cls._registry.keys())

    @classmethod
    def create(cls, item_type: Union[str, Models]) -> Node:
        """Create a item instance of the specified type.

        Args:
            item_type: Type identifier of item to create

        Returns:
            Instance of requested item class

        Raises:
            KeyError: If item_type is not registered
        """
        try:
            item_class = cls._registry[item_type]
            return item_class
        except KeyError:
            available = ', '.join(cls._registry.keys())
            raise KeyError(
                f"Unknown item type: {item_type}. "
                f"Available types: {available}"
            )

    @classmethod
    def register(cls, item_type, item_class) -> None:
        """Register a new item type.

        Args:
            item_type: Unique identifier for the item
            item_class: Node class to register, must inherit from Node

        Raises:
            TypeError: If item_class doesn't inherit from Node
            ValueError: If item_type is already registered
        """
        if not issubclass(item_class, cls.get_base_type()):
            raise TypeError(
                f"Class must inherit from {cls.get_base_type()}: {item_class}"
            )
        if item_type in cls._registry:
            raise ValueError(f"Type already registered: {item_type}")
        cls._registry[item_type] = item_class

    @classmethod
    def get_base_type(cls):
        """Get the base type for factory products.

        Must be implemented by subclasses.

        Returns:
            Base type that all products must inherit from
        """
        raise NotImplementedError


class NodeFactory(BaseFactory):
    """Implementation of Factory Node class.
    """
    _registry: Dict[Models, Type[Node]] = {
        Models.BASE: Node,
        Models.RELAY: RelayNode,
        Models.LINEAR: LinearNode,
        Models.RELU: ReLUNode,
        Models.OSCILLATOR: OscillatorNode,
        Models.HOPF_OSCILLATOR: HopfOscillatorNode,
        # Models.MORPHED_OSCILLATOR: MorphedOscillatorNode,
        # Models.MATSUOKA: MatsuokaNode,
        # Models.FITZHUGH_NAGUMO: FitzhughNagumoNode,
        # Models.MORRIS_LECAR: MorrisLecarNode,
        # Models.LEAKY_INTEGRATOR: LeakyIntegratorNode,
        Models.LI_DANNER: LIDannerNode,
        Models.LI_NAP_DANNER: LINaPDannerNode,
        # Models.LI_DAUN: LIDaunNode,
        # Models.HH_DAUN: HHDaunNode,
    }

    @classmethod
    def get_base_type(cls) -> Type[Node]:
        return Node


class EdgeFactory(BaseFactory):
    """Implementation of Factory Edge class."""
    _registry: Dict[Models, Type[Edge]] = {
        Models.BASE: Edge,
        Models.OSCILLATOR: OscillatorEdge,
    }

    @classmethod
    def get_base_type(cls) -> Type[Edge]:
        return Edge
