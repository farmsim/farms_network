""" Node """


from abc import ABC, abstractmethod
from typing import Any, Dict, Type

from farms_network.core.node_cy import NodeCy
from farms_network.core.options import NodeOptions


class Node(ABC):

    CY_NODE_CLASS: Type[NodeCy] = None

    def __init__(self, name: str, model: str, **parameters):
        self.name: str = name        # Unique name of the node
        self.model: str = model      # Type of the model (e.g., "empty")
        self._node_cy = self._create_cy_node(**parameters)

    def _create_cy_node(self, **kwargs) -> NodeCy:
        if self.CY_NODE_CLASS is None:
            raise NotImplementedError("Must define CY_NODE_CLASS")
        return self.CY_NODE_CLASS(**kwargs)

    # General node properties
    @property
    def nstates(self):
        return self._node_cy.nstates

    @property
    def nparams(self):
        return self._node_cy.nparams

    @property
    def ninputs(self):
        return self._node_cy.ninputs

    @property
    def is_statefull(self):
        return self._node_cy.is_statefull

    def print_parameters(self):
        return self._node_cy.parameters

    def input_tf(self):
        """ Input transfer function """
        pass

    def ode(self, time, states, derivatives, external_input, network_outputs, inputs, weights, noise):
        """ ODE computation """
        return self._node_cy.ode(
            time, states, derivatives, external_input,
            network_outputs, inputs, weights, noise
        )

    def output_tf(self, time, states, input_val, noise):
        """ ODE computation """
        return self._node_cy.output_tf(time, states, input_val, noise)

    @classmethod
    def from_options(cls, node_options: NodeOptions):
        """ From node options """
        name: str = node_options.name
        parameters = node_options.parameters
        if parameters is None:
            parameters = {}
        return cls(name, **parameters)

    def to_options(self):
        """ To node options """
        name: str = node_options.name
        parameters = node_options.parameters
        return cls(name, **parameters)

    def debug_info(self):
        """ Get debug information about the node """
        return {
            'class': self.__class__.__name__,
            'model': self.model,
            'name': self.name,
            'nstates': self.nstates,
            'ninputs': self.ninputs,
            'nparams': self.nparams,
            'is_statefull': self.is_statefull,
            'initialized': self._initialized,
            'has_ode_func': self._node.ode_func is not NULL,
            'has_output_func': self._node.output_func is not NULL,
            'has_params': self._node.params is not NULL,
            'parameters': self.parameters
        }
