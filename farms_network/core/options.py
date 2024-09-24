""" Options to configure the neural and network models """

from typing import List, Self

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

    def __add__(self, other: Self):
        """ Combine two network options """
        assert isinstance(other, NetworkOptions)
        for node in other.nodes:
            self.add_node(node)
        for edge in other.edges:
            self.add_edge(edge)
        return self


###########################
# Node Base Class Options #
###########################
class NodeOptions(Options):
    """ Base class for defining node options """

    def __init__(self, **kwargs):
        """ Initialize """
        super().__init__()
        self.name: str = kwargs.pop("name")

        self.model: str = kwargs.pop("model")
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
        self.names: List[str] = kwargs.pop("names")
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')


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

    def __eq__(self, other):
        if isinstance(other, EdgeOptions):
            return (
                (self.source == other.source) and
                (self.target == other.target)
            )
        return False


class EdgeVisualOptions(Options):
    """ Base class for edge visualization parameters """

    def __init__(self, **kwargs):
        super().__init__()
        self.color: List[float] = kwargs.pop("color", [1.0, 0.0, 0.0])
        self.label: str = kwargs.pop("label", "")
        self.layer: str = kwargs.pop("layer", "background")
        self.latex: dict = kwargs.pop("latex", "{}")


#########################################
# Leaky Integrator Danner Model Options #
#########################################
class LIDannerNodeOptions(NodeOptions):
    """ Class to define the properties of Leaky integrator danner node model """

    def __init__(self, **kwargs):
        """ Initialize """
        model = "li_danner"
        super().__init__(
            name=kwargs.pop("name"),
            model=model,
            parameters=kwargs.pop("parameters"),
            visual=kwargs.pop("visual"),
            state=kwargs.pop("state"),
        )
        self._nstates = 2
        self._nparameters = 13

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')


class LIDannerParameterOptions(NodeParameterOptions):
    """
    Class to define the parameters of Leaky Integrator Danner node model.

    Attributes:
        c_m (float): Membrane capacitance (in pF).
        g_leak (float): Leak conductance (in nS).
        e_leak (float): Leak reversal potential (in mV).
        v_max (float): Maximum voltage (in mV).
        v_thr (float): Threshold voltage (in mV).
        g_syn_e (float): Excitatory synaptic conductance (in nS).
        g_syn_i (float): Inhibitory synaptic conductance (in nS).
        e_syn_e (float): Excitatory synaptic reversal potential (in mV).
        e_syn_i (float): Inhibitory synaptic reversal potential (in mV).
    """

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
    def defaults(cls, **kwargs):
        """ Get the default parameters for LI Danner Node model """

        options = {}

        options["c_m"] = kwargs.pop("c_m",  10.0)
        options["g_leak"] = kwargs.pop("g_leak",  2.8)
        options["e_leak"] = kwargs.pop("e_leak",  -60.0)
        options["v_max"] = kwargs.pop("v_max",  0.0)
        options["v_thr"] = kwargs.pop("v_thr",  -50.0)
        options["g_syn_e"] = kwargs.pop("g_syn_e",  10.0)
        options["g_syn_i"] = kwargs.pop("g_syn_i",  10.0)
        options["e_syn_e"] = kwargs.pop("e_syn_e",  -10.0)
        options["e_syn_i"] = kwargs.pop("e_syn_i",  -75.0)

        return cls(**options)


class LIDannerStateOptions(NodeStateOptions):
    """ LI Danner node state options """

    STATE_NAMES = ["v0",]

    def __init__(self, **kwargs):
        super().__init__(
            initial=kwargs.pop("initial"),
            names=LIDannerStateOptions.STATE_NAMES
        )
        assert len(self.initial) == 1, f"Number of initial states {len(self.initial)} should be 1"

    @classmethod
    def from_kwargs(cls, **kwargs):
        """ From node specific name-value kwargs """
        initial = [
            kwargs.pop(name)
            for name in cls.STATE_NAMES
        ]
        return cls(initial=initial)


##################################################
# Leaky Integrator With NaP Danner Model Options #
##################################################
class LIDannerNaPNodeOptions(NodeOptions):
    """ Class to define the properties of Leaky integrator danner node model """

    def __init__(self, **kwargs):
        """ Initialize """
        model = "li_danner_nap"
        super().__init__(
            name=kwargs.pop("name"),
            model=model,
            parameters=kwargs.pop("parameters"),
            visual=kwargs.pop("visual"),
            state=kwargs.pop("state"),
        )
        self._nstates = 2
        self._nparameters = 19

        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')


class LIDannerNaPParameterOptions(NodeParameterOptions):
    """ Class to define the parameters of Leaky integrator danner node model """

    def __init__(self, **kwargs):
        super().__init__()

        self.c_m = kwargs.pop("c_m")                        # pF
        self.g_nap = kwargs.pop("g_nap")                    # nS
        self.e_na = kwargs.pop("e_na")                      # mV
        self.v1_2_m = kwargs.pop("v1_2_m")                  # mV
        self.k_m = kwargs.pop("k_m")                        #
        self.v1_2_h = kwargs.pop("v1_2_h")                  # mV
        self.k_h = kwargs.pop("k_h")                        #
        self.v1_2_t = kwargs.pop("v1_2_t")                  # mV
        self.k_t = kwargs.pop("k_t")                        #
        self.g_leak = kwargs.pop("g_leak")                  # nS
        self.e_leak = kwargs.pop("e_leak")                  # mV
        self.tau_0 = kwargs.pop("tau_0")                    # mS
        self.tau_max = kwargs.pop("tau_max")                # mS
        self.v_max = kwargs.pop("v_max")                    # mV
        self.v_thr = kwargs.pop("v_thr")                    # mV
        self.g_syn_e = kwargs.pop("g_syn_e")                # nS
        self.g_syn_i = kwargs.pop("g_syn_i")                # nS
        self.e_syn_e = kwargs.pop("e_syn_e")                # mV
        self.e_syn_i = kwargs.pop("e_syn_i")                # mV

    @classmethod
    def defaults(cls, **kwargs):
        """ Get the default parameters for LI NaP Danner Node model """

        options = {}

        options["c_m"] = kwargs.pop("c_m", 10.0)                  # pF
        options["g_nap"] = kwargs.pop("g_nap", 4.5)               # nS
        options["e_na"] = kwargs.pop("e_na", 50.0)                # mV
        options["v1_2_m"] = kwargs.pop("v1_2_m", -40.0)           # mV
        options["k_m"] = kwargs.pop("k_m", -6.0)                  #
        options["v1_2_h"] = kwargs.pop("v1_2_h", -45.0)           # mV
        options["k_h"] = kwargs.pop("k_h", 4.0)                   #
        options["v1_2_t"] = kwargs.pop("v1_2_t", -35.0)           # mV
        options["k_t"] = kwargs.pop("k_t", 15.0)                  #
        options["g_leak"] = kwargs.pop("g_leak", 4.5)             # nS
        options["e_leak"] = kwargs.pop("e_leak", -62.5)           # mV
        options["tau_0"] = kwargs.pop("tau_0", 80.0)              # mS
        options["tau_max"] = kwargs.pop("tau_max", 160.0)         # mS
        options["v_max"] = kwargs.pop("v_max", 0.0)               # mV
        options["v_thr"] = kwargs.pop("v_thr", -50.0)             # mV
        options["g_syn_e"] = kwargs.pop("g_syn_e", 10.0)          # nS
        options["g_syn_i"] = kwargs.pop("g_syn_i", 10.0)          # nS
        options["e_syn_e"] = kwargs.pop("e_syn_e", -10.0)         # mV
        options["e_syn_i"] = kwargs.pop("e_syn_i", -75.0)         # mV

        return cls(**options)


class LIDannerNaPStateOptions(NodeStateOptions):
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
        parameters=LIDannerNaPParameterOptions.defaults(),
        visual=NodeVisualOptions(),
        state=LIDannerNaPStateOptions.from_kwargs(
            v0=-60.0, h0=0.1
        ),
    )

    print(f"Is hashable {isinstance(li_opts, typing.Hashable)}")

    network_opts.add_node(li_opts)
    li_opts = LIDannerNaPNodeOptions(
        name="li-2",
        parameters=LIDannerNaPParameterOptions.defaults(),
        visual=NodeVisualOptions(),
        state=LIDannerNaPStateOptions.from_kwargs(
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

    network = NetworkOptions.load("/tmp/rhythm_opts.yaml")

    graph = nx.node_link_graph(
        network,
        directed=True,
        multigraph=False,
        link="edges",
        name="name",
        source="from_node",
        target="to_node"
    )
    nx.draw(
        graph, pos=nx.nx_agraph.graphviz_layout(graph),
        node_shape="s",
        connectionstyle="arc3,rad=-0.2"
    )
    plt.show()


if __name__ == '__main__':
    main()
