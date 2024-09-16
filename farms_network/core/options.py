""" Options to configure the neural and network models """

from typing import List

import matplotlib.pyplot as plt
import networkx as nx
from farms_core.options import Options


##############################
# Network Base Class Options #
##############################
class NetworkOptions(Options):
    """ Base class for neural network options """

    def __init__(self, **kwargs):
        super().__init__()
        _name: str = kwargs.pop("name", "")
        # Default properties to make it compatible with networkx
        self.directed: bool = kwargs.pop("directed", True)
        self.multigraph: bool = kwargs.pop("multigraph", False)
        self.graph: dict = {
            "name": _name
        }

        self.units = None

        self.nodes: List[NodeOptions] = kwargs.pop("nodes", [])
        self.edges: List[EdgeOptions] = kwargs.pop("edges", [])

    def add_node(self, options: "NodeOptions"):
        """ Add a node if it does not already exist in the list """
        assert isinstance(options, NodeOptions), f"{type(options)} not an instance of NodeOptions"
        if options not in self.nodes:
            self.nodes.append(options)
        else:
            print(f"Node {options.name} already exists and will not be added again.")

    def add_edge(self, options: "EdgeOptions"):
        """ Add a node if it does not already exist in the list """
        if (options.from_node in self.nodes) and (options.to_node in self.nodes):
            self.edges.append(options)
        else:
            print(f"Edge {options} does not contain the nodes.")


###########################
# Node Base Class Options #
###########################
class NodeOptions(Options):
    """ Base class for defining node options """

    def __init__(self, **kwargs):
        """ Initialize """
        super().__init__()
        self.name: str = kwargs.pop("name")

        self.parameters: NodeParameterOptions = kwargs.pop("parameters")
        self.visual: NodeVisualOptions = kwargs.pop("visual")
        self.state: NodeStateOptions = kwargs.pop("state")

        self._nstates: int = 0
        self._nparameters: int = 0

    def __eq__(self, other):
        if isinstance(other, NodeOptions):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return False

    def __hash__(self):
        return hash(self.name)  # Hash based on the node name (or any unique property)


class NodeParameterOptions(Options):
    """ Base class for node specific parameters """

    def __init__(self):
        super().__init__()


class NodeStateOptions(Options):
    """ Base class for node specific state options """

    def __init__(self, **kwargs):
        super().__init__()
        self.initial: List[float] = kwargs.pop("initial")


class NodeVisualOptions(Options):
    """ Base class for node visualization parameters """

    def __init__(self, **kwargs):
        super().__init__()
        self.position: List[float] = kwargs.pop("position", [0.0, 0.0, 0.0])
        self.radius: float = kwargs.pop("radius", 1.0)
        self.color: List[float] = kwargs.pop("color", [1.0, 0.0, 0.0])
        self.label: str = kwargs.pop("label", "n")
        self.layer: str = kwargs.pop("layer", "background")
        self.latex: dict = kwargs.pop("latex", "{}")


################
# Edge Options #
################
class EdgeOptions(Options):
    """ Base class for defining edge options between nodes """

    def __init__(self, **kwargs):
        """ Initialize """
        super().__init__()

        self.from_node: str = kwargs.pop("from_node")
        self.to_node: str = kwargs.pop("to_node")
        self.weight: float = kwargs.pop("weight")
        self.type: str = kwargs.pop("type")

        self.visual: NodeVisualOptions = kwargs.pop("visual")


class EdgeVisualOptions(Options):
    """ Base class for edge visualization parameters """

    def __init__(self, **kwargs):
        super().__init__()
        self.position: List[float] = kwargs.pop("position", [0.0, 0.0, 0.0])
        self.radius: float = kwargs.pop("radius", 1.0)
        self.color: List[float] = kwargs.pop("color", [1.0, 0.0, 0.0])
        self.label: str = kwargs.pop("label", "n")
        self.layer: str = kwargs.pop("layer", "background")
        self.latex: dict = kwargs.pop("latex", "{}")


#########################################
# Leaky Integrator Danner Model Options #
#########################################
class LIDannerNodeOptions(NodeOptions):
    """ Class to define the properties of Leaky integrator danner node model """

    def __init__(self, **kwargs):
        """ Initialize """
        super().__init__(
            name=kwargs.pop("name"),
            parameters=kwargs.pop("parameters"),
            visual=kwargs.pop("visual"),
            state=kwargs.pop("state"),
        )
        self._nstates = 2
        self._nparameters = 13

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')


class LIDannerParameterOptions(NodeParameterOptions):
    """ Class to define the parameters of Leaky integrator danner node model """

    def __init__(self, **kwargs):
        super().__init__()
        self.c_m = kwargs.pop("c_m")                        # pF
        self.g_leak = kwargs.pop("g_leak")                  # nS
        self.e_leak = kwargs.pop("e_leak")                  # mV
        self.v_max = kwargs.pop("v_max")                    # mV
        self.v_thr = kwargs.pop("v_thr")                    # mV
        self.g_syn_e = kwargs.pop("g_syn_e")                # nS
        self.g_syn_i = kwargs.pop("g_syn_i")                # nS
        self.e_syn_e = kwargs.pop("e_syn_e")                # mV
        self.e_syn_i = kwargs.pop("e_syn_i")                # mV

    @classmethod
    def defaults(cls):
        """ Get the default parameters for LI Danner Node model """

        options = {}

        options["c_m"] = 10.0
        options["g_leak"] = 2.8
        options["e_leak"] = -60.0
        options["v_max"] = 0.0
        options["v_thr"] = -50.0
        options["g_syn_e"] = 10.0
        options["g_syn_i"] = 10.0
        options["e_syn_e"] = -10.0
        options["e_syn_i"] = -75.0

        return cls(**options)


class LIDannerStateOptions(NodeStateOptions):
    """ LI Danner node state options """

    def __init__(self, **kwargs):
        super().__init__(
            initial=kwargs.pop("initial")
        )
        assert len(self.initial) == 2, f"Number of initial states {len(self.initial)} should be 2"

    @classmethod
    def from_kwargs(cls, **kwargs):
        """ From node specific name-value kwargs """
        v0 = kwargs.pop("v0")
        h0 = kwargs.pop("h0")
        initial = [v0, h0]
        return cls(initial=initial)


##################################################
# Leaky Integrator With NaP Danner Model Options #
##################################################
class LIDannerNodeOptions(NodeOptions):
    """ Class to define the properties of Leaky integrator danner node model """

    def __init__(self, **kwargs):
        """ Initialize """
        super().__init__(
            name=kwargs.pop("name"),
            parameters=kwargs.pop("parameters"),
            visual=kwargs.pop("visual"),
            state=kwargs.pop("state"),
        )
        self._nstates = 2
        self._nparameters = 9

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')


class LIDannerParameterOptions(NodeParameterOptions):
    """ Class to define the parameters of Leaky integrator danner node model """

    def __init__(self, **kwargs):
        super().__init__()
        self.c_m = kwargs.pop("c_m")                        # pF
        self.g_leak = kwargs.pop("g_leak")                  # nS
        self.e_leak = kwargs.pop("e_leak")                  # mV
        self.v_max = kwargs.pop("v_max")                    # mV
        self.v_thr = kwargs.pop("v_thr")                    # mV
        self.g_syn_e = kwargs.pop("g_syn_e")                # nS
        self.g_syn_i = kwargs.pop("g_syn_i")                # nS
        self.e_syn_e = kwargs.pop("e_syn_e")                # mV
        self.e_syn_i = kwargs.pop("e_syn_i")                # mV

    @classmethod
    def defaults(cls):
        """ Get the default parameters for LI Danner Node model """

        options = {}

        options["c_m"] = 10.0
        options["g_leak"] = 2.8
        options["e_leak"] = -60.0
        options["v_max"] = 0.0
        options["v_thr"] = -50.0
        options["g_syn_e"] = 10.0
        options["g_syn_i"] = 10.0
        options["e_syn_e"] = -10.0
        options["e_syn_i"] = -75.0

        return cls(**options)


class LIDannerStateOptions(NodeStateOptions):
    """ LI Danner node state options """

    def __init__(self, **kwargs):
        super().__init__(
            initial=kwargs.pop("initial")
        )
        assert len(self.initial) == 2, f"Number of initial states {len(self.initial)} should be 2"

    @classmethod
    def from_kwargs(cls, **kwargs):
        """ From node specific name-value kwargs """
        v0 = kwargs.pop("v0")
        h0 = kwargs.pop("h0")
        initial = [v0, h0]
        return cls(initial=initial)



########
# MAIN #
########
def main():
    """ Main """

    import typing

    network_opts = NetworkOptions(name="new-network")

    li_opts = LIDannerNodeOptions(
        name="li",
        parameters=LIDannerParameterOptions.defaults(),
        visual=NodeVisualOptions(),
        state=LIDannerStateOptions.from_kwargs(
            v0=-60.0, h0=0.1
        ),
    )

    print(f"Is hashable {isinstance(li_opts, typing.Hashable)}")

    network_opts.add_node(li_opts)
    li_opts = LIDannerNodeOptions(
        name="li-2",
        parameters=LIDannerParameterOptions.defaults(),
        visual=NodeVisualOptions(),
        state=LIDannerStateOptions.from_kwargs(
            v0=-60.0, h0=0.1
        ),
    )
    network_opts.add_node(li_opts)

    network_opts.add_edge(
        EdgeOptions(
            from_node="li",
            to_node="li-2",
            weight=0.0,
            type="excitatory",
            visual=EdgeVisualOptions()
        )
    )

    network_opts.save("/tmp/network_opts.yaml")

    network = NetworkOptions.load("/tmp/network_opts.yaml")

    graph = nx.node_link_graph(
        network,
        directed=True,
        multigraph=False,
        link="edges",
        name="name",
        source="from_node",
        target="to_node"
    )
    nx.draw(graph)
    plt.show()

if __name__ == '__main__':
    main()
