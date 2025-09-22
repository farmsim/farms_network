""" Generate and reproduce Zhang, Shevtsova, et al. eLife 2022;11:e73424. DOI:
https://doi.org/10.7554/eLife.73424 paper network """


import os
from copy import deepcopy
from pprint import pprint

import farms_pylog as pylog
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from farms_core.io.yaml import read_yaml, write_yaml
from farms_core.options import Options
from farms_network.core import options
from farms_network.core.data import (NetworkConnectivity, NetworkData,
                                     NetworkStates)
from farms_network.core.network import PyNetwork, rk4
from farms_network.core.options import NetworkOptions
from scipy.integrate import ode
from tqdm import tqdm


class RhythmGenerator:
    """Generate RhythmGenerator Network"""

    def __init__(self, name="", anchor_x=0.0, anchor_y=0.0):
        """Initialization."""
        super().__init__()
        self.name = name

    def nodes(self):
        """Add nodes."""
        nodes = {}
        nodes["RG-F"] = options.LINaPDannerNodeOptions(
            name=self.name + "-RG-F",
            parameters=options.LINaPDannerParameterOptions.defaults(),
            visual=options.NodeVisualOptions(label="F", color=[1.0, 0.0, 0.0]),
            state=options.LINaPDannerStateOptions.from_kwargs(
                v0=-62.5, h0=np.random.uniform(0, 1)
            ),
        )

        nodes["RG-E"] = options.LINaPDannerNodeOptions(
            name=self.name + "-RG-E",
            parameters=options.LINaPDannerParameterOptions.defaults(),
            visual=options.NodeVisualOptions(label="E", color=[0.0, 1.0, 0.0]),
            state=options.LINaPDannerStateOptions.from_kwargs(
                v0=-62.5, h0=np.random.uniform(0, 1)
            ),
        )

        nodes["In-F"] = options.LIDannerNodeOptions(
            name=self.name + "-In-F",
            parameters=options.LIDannerParameterOptions.defaults(),
            visual=options.NodeVisualOptions(label="In", color=[0.2, 0.2, 0.2]),
            state=options.LIDannerStateOptions.from_kwargs(v0=-60.0,),
        )

        nodes["In-E"] = options.LIDannerNodeOptions(
            name=self.name + "-In-E",
            parameters=options.LIDannerParameterOptions.defaults(),
            visual=options.NodeVisualOptions(label="In", color=[0.2, 0.2, 0.2]),
            state=options.LIDannerStateOptions.from_kwargs(v0=-60.0,),
        )
        return nodes

    def edges(self):
        edges = {}
        edges["RG-F-to-In-F"] = options.EdgeOptions(
            source=self.name + "-RG-F",
            target=self.name + "-In-F",
            weight=0.4,
            type="excitatory",
            visual=options.EdgeVisualOptions(),
        )
        edges["In-F-to-RG-E"] = options.EdgeOptions(
            source=self.name + "-In-F",
            target=self.name + "-RG-E",
            weight=-1.0,
            type="inhibitory",
            visual=options.EdgeVisualOptions(),
        )
        edges["RG-E-to-In-E"] = options.EdgeOptions(
            source=self.name + "-RG-E",
            target=self.name + "-In-E",
            weight=0.4,
            type="excitatory",
            visual=options.EdgeVisualOptions(),
        )
        edges["In-E-to-RG-F"] = options.EdgeOptions(
            source=self.name + "-In-E",
            target=self.name + "-RG-F",
            weight=-0.08,
            type="inhibitory",
            visual=options.EdgeVisualOptions(),
        )
        return edges


class Commissural(nx.DiGraph):
    """Commissural Network template."""


    def __init__(self, name="", anchor_x=0.0, anchor_y=0.0):
        """Initialization."""
        super().__init__(name=name)

    def nodes(self):
        """Add nodes."""
        nodes = {}
        # V3
        node[] = options.LINaPDannerNodeOptions(
            name=self.name + "-RG-F",
            parameters=options.LINaPDannerParameterOptions.defaults(),
            visual=options.NodeVisualOptions(label="F", color=[1.0, 0.0, 0.0]),
            state=options.LINaPDannerStateOptions.from_kwargs(
                v0=-62.5, h0=np.random.uniform(0, 1)
            ),
        )

        nodes[self.name + "-V3-E-Left-Fore"] = options.LIDannerNodeOptions(
            name=self.name + "-V3-E-Left-Fore",
            parameters=options.LIDannerParameterOptions.defaults(),
            visual=options.NodeVisualOptions(label="In", color=[0.2, 0.2, 0.2]),
            state=options.LIDannerStateOptions.from_kwargs(v0=-60.0,),
        )

        self.add_node(
            self.name + "-V3-E-Right-Fore",
            label="V3-E",
            model="lif_danner",
            x=2.0 + anchor_x,
            y=8.0 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.add_node(
            self.name + "-V3-E-Left-Hind",
            label="V3-E",
            model="lif_danner",
            x=-1.0 + anchor_x,
            y=-6.0 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.add_node(
            self.name + "-V3-E-Right-Hind",
            label="V3-E",
            model="lif_danner",
            x=2.0 + anchor_x,
            y=-6.0 + anchor_y,
            color="g",
            v0=-60.0,
        )

        self.add_node(
            self.name + "-V0-V-Left-Fore",
            label="$V0_V$",
            model="lif_danner",
            x=-1.0 + anchor_x,
            y=6.0 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.add_node(
            self.name + "-V0-V-Right-Fore",
            label="$V0_V$",
            model="lif_danner",
            x=2.0 + anchor_x,
            y=6.0 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.add_node(
            self.name + "-V0-V-Left-Hind",
            label="$V0_V$",
            model="lif_danner",
            x=-1.0 + anchor_x,
            y=-4.0 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.add_node(
            self.name + "-V0-V-Right-Hind",
            label="$V0_V$",
            model="lif_danner",
            x=2.0 + anchor_x,
            y=-4.0 + anchor_y,
            color="g",
            v0=-60.0,
        )

        self.add_node(
            self.name + "-V0-D-Left-Fore",
            label="$V0_D$",
            model="lif_danner",
            x=-1.0 + anchor_x,
            y=4.0 + anchor_y,
            colom="g",
            v0=-60.0,
        )
        self.add_node(
            self.name + "-V0-D-Right-Fore",
            label="$V0_D$",
            model="lif_danner",
            x=2.0 + anchor_x,
            y=4.0 + anchor_y,
            color="m",
            v0=-60.0,
        )
        self.add_node(
            self.name + "-V0-D-Left-Hind",
            label="$V0_D$",
            model="lif_danner",
            x=-1.0 + anchor_x,
            y=-2.0 + anchor_y,
            color="m",
            v0=-60.0,
        )
        self.add_node(
            self.name + "-V0-D-Right-Hind",
            label="$V0_D$",
            model="lif_danner",
            x=2.0 + anchor_x,
            y=-2.0 + anchor_y,
            color="m",
            v0=-60.0,
        )

        self.add_node(
            self.name + "-V3-F-Left-Fore",
            label="V3-F",
            model="lif_danner",
            x=-1.0 + anchor_x,
            y=2.0 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.add_node(
            self.name + "-V3-F-Right-Fore",
            label="V3-F",
            model="lif_danner",
            x=2.0 + anchor_x,
            y=2.0 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.add_node(
            self.name + "-V3-F-Left-Hind",
            label="V3-F",
            model="lif_danner",
            x=-1.0 + anchor_x,
            y=-0.0 + anchor_y,
            color="g",
            v0=-60.0,
        )

        self.add_node(
            self.name + "-V3-F-Right-Hind",
            label="V3-F",
            model="lif_danner",
            x=2.0 + anchor_x,
            y=-0.0 + anchor_y,
            color="g",
            v0=-60.0,
        )

        self.add_node(
            self.name + "-V2a-diag",
            label="$V2_a$",
            x=(-1.0 if self.name[-1] == "L" else 3.0) + anchor_x,
            model="lif_danner",
            y=6.0 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.add_node(
            self.name + "-CINi1",
            label="i1",
            model="lif_danner",
            x=1.0 + anchor_x,
            y=-1.0 + anchor_y,
            color="m",
            v0=-60.0,
        )
        self.add_node(
            self.name + "-V0V",
            label="$V0_V$",
            model="lif_danner",
            x=1.0 + anchor_x,
            y=1.0 + anchor_y,
            color="g",
            m_i=0.15,
            b_i=0.0,
            v0=-60.0,
        )
        self.add_node(
            self.name + "-V0D",
            label="$V0_D$",
            model="lif_danner",
            x=1.0 + anchor_x,
            y=3.0 + anchor_y,
            color="m",
            m_i=0.75,
            b_i=0.0,
            v0=-60.0,
        )
        self.add_node(
            self.name + "-V2a",
            label="$V2a$",
            model="lif_danner",
            x=(-1.0 if self.name[-1] == "L" else 3.0) + anchor_x,
            y=1.0 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.add_node(
            self.name + "-IniV0V",
            label="$V0_V$",
            model="lif_danner",
            x=(-1.0 if self.name[-1] == "L" else 3.0) + anchor_x,
            y=3 + anchor_y,
            color="g",
            v0=-60.0,
        )

    def add_connections(self):
        """ Add local connections """
        ...


class LPSN(nx.DiGraph):
    """Long Propriospinal Neuron Network template."""

    def __init__(self, name="", anchor_x=0.0, anchor_y=0.0):
        """Initialization."""
        super().__init__(name=name)

    @classmethod
    def generate_nodes_edges(cls, name, anchor_x=0.0, anchor_y=0.0):
        obj = cls(name, anchor_x, anchor_y)
        obj.add_neurons(anchor_x, anchor_y)
        obj.add_connections()
        return obj

    def add_neurons(self, anchor_x, anchor_y):
        """Add neurons."""
        self.add_node(
            self.name + "-V0D-diag",
            label="$V0_D$",
            model="lif_danner",
            x=1.0 + anchor_x,
            y=0.0 + anchor_y,
            color="m",
            m_i=0.75,
            b_i=0.0,
            v0=-60.0,
        )
        self.add_node(
            self.name + "-V0V-diag-fh",
            label="$V0_V$",
            model="lif_danner",
            x=1.0 + anchor_x,
            y=2 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.add_node(
            self.name + "-V0V-diag-hf",
            label="$V0_V$",
            model="lif_danner",
            x=1.0 + anchor_x,
            y=4 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.add_node(
            self.name + "-Ini-hom-fh",
            label="$In_i$",
            model="lif_danner",
            x=(-2.0 if self.name[-1] == "L" else 4.0) + anchor_x,
            y=4 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.add_node(
            self.name + "-Sh2-hom-fh",
            label="$Sh_2$",
            model="lif_danner",
            x=(-4.0 if self.name[-1] == "L" else 6.0) + anchor_x,
            y=0 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.add_node(
            self.name + "-Sh2-hom-hf",
            label="$Sh_2$",
            model="lif_danner",
            x=(-4.0 if self.name[-1] == "L" else 6.0) + anchor_x,
            y=4 + anchor_y,
            color="g",
            v0=-60.0,
        )

    def add_connections(self):
        """ Add local connections """
        ...


_DOC_WRAPPER_TIKZ = r"""\documentclass{{article}}
\usepackage{{tikz}}
\usetikzlibrary{{
  arrows,
  arrows.meta,
  calc,
  backgrounds,
  fit,
  positioning,
  shapes
}}
\usepackage{{subcaption}}

\begin{{document}}
\definecolor{{flex}}{{RGB}}{{190,174,212}}
\definecolor{{inf}}{{RGB}}{{253,192,134}}
\definecolor{{ext}}{{RGB}}{{127,201,127}}

\tikzset{{
% Color shades
my-node/.style={{
  circle, minimum size=1.0 cm, inner sep=0.04cm, outer sep=0.04cm, draw=black, thick, double, font={{\small}}
}},
}}
{content}
\end{{document}}"""


def generate_network():
    """ Generate network """

    # Main network
    network_options = options.NetworkOptions(
        directed=True,
        multigraph=False,
        graph={"name": "zhang2022"},
    )

    # Generate rhythm centers
    for side in ("LEFT", "RIGHT"):
        for limb in ("FORE", "HIND"):
            rhythm = RhythmGenerator(name=f"{side}-{limb}")
            network_options.add_nodes((rhythm.nodes()).values())
            network_options.add_edges((rhythm.edges()).values())

    flexor_drive = options.LinearNodeOptions(
        name="FD",
        parameters=options.LinearParameterOptions.defaults(slope=0.1, bias=0.0),
        visual=options.NodeVisualOptions(),
    )
    extensor_drive = options.LinearNodeOptions(
        name="ED",
        parameters=options.LinearParameterOptions.defaults(slope=0.0, bias=0.1),
        visual=options.NodeVisualOptions(),
    )
    network_options.add_node(flexor_drive)
    network_options.add_node(extensor_drive)
    for side in ("LEFT", "RIGHT"):
        for limb in ("FORE", "HIND"):
            network_options.add_edge(
                options.EdgeOptions(
                    source="FD",
                    target=f"{side}-{limb}-RG-F",
                    weight=1.0,
                    type="excitatory",
                    visual=options.EdgeVisualOptions(),
                )
            )
            network_options.add_edge(
                options.EdgeOptions(
                    source="ED",
                    target=f"{side}-{limb}-RG-E",
                    weight=1.0,
                    type="excitatory",
                    visual=options.EdgeVisualOptions(),
                )
            )

    data = NetworkData.from_options(network_options)

    network = PyNetwork.from_options(network_options)

    # nnodes = len(network_options.nodes)
    # integrator.set_initial_value(np.zeros(len(data.states.array),), 0.0)

    # print("Data ------------", np.array(network.data.states.array))

    # data.to_file("/tmp/sim.hdf5")

    # integrator.integrate(integrator.t + 1e-3)

    # # Integrate
    iterations = 10000
    states = np.ones((iterations+1, len(data.states.array)))*1.0
    outputs = np.ones((iterations, len(data.outputs.array)))*1.0
    # states[0, 2] = -1.0

    for iteration in tqdm(range(0, iterations), colour='green', ascii=' >='):
        network.data.external_inputs.array[:] = np.ones((1,))*(iteration/iterations)*1.0
        states[iteration+1, :] = rk4(iteration*1e-3, states[iteration, :], network.ode, step_size=1)
        outputs[iteration, :] = network.data.outputs.array

    plt.figure()
    plt.fill_between(
        np.linspace(0.0, iterations*1e-3, iterations), outputs[:, 0],
        alpha=0.2, lw=1.0,
    )
    plt.plot(
        np.linspace(0.0, iterations*1e-3, iterations), outputs[:, 0],
        label="RG-F"
    )
    plt.fill_between(
        np.linspace(0.0, iterations*1e-3, iterations), outputs[:, 1],
        alpha=0.2, lw=1.0,
    )
    plt.plot(np.linspace(0.0, iterations*1e-3, iterations), outputs[:, 1], label="RG-E")
    plt.legend()

    network_options.save("/tmp/netwok_options.yaml")

    graph = nx.node_link_graph(
        network_options,
        directed=True,
        multigraph=False,
        link="edges",
        name="name",
        source="source",
        target="target"
    )
    plt.figure()
    nx.draw(
        graph, pos=nx.nx_agraph.graphviz_layout(graph),
        node_shape="o",
        connectionstyle="arc3,rad=-0.2",
        with_labels=True,
    )
    plt.show()

    # # Commissural
    # commissural = Commissural.generate_nodes_edges(name="commissural")

    # # LSPN
    # lpsn = LPSN.generate_nodes_edges(name="lspn")

    # network = nx.compose_all(
    #     [rhythm1, rhythm2, rhythm3, rhythm4, commissural, lpsn]
    # )

    # nx.write_graphml(network, "./config/auto_zhang_2022.graphml")
    # nx.write_latex(
    #     network,
    #     "zhang2022_figure.tex",
    #     pos={name: (node['x'], node['y']) for name, node in network.nodes.items()},
    #     as_document=True,
    #     caption="A path graph",
    #     latex_label="fig1",
    #     node_options=node_options,
    #     default_node_options="my-node",
    #     node_label={
    #         name: node["label"]
    #         for name, node in network.nodes.items()
    #     },
    #     default_edge_options="[color=black, thick, -{Latex[scale=1.0]}, bend left, looseness=0.75]",
    #     document_wrapper=_DOC_WRAPPER_TIKZ,
    # )

    # latex_code = nx.to_latex(network)  # a string rather than a file

    # os.system("pdflatex zhang2022_figure.tex")


def main():
    """Main."""

    # Generate the network
    generate_network()

    # Run the network
    # run_network()


if __name__ == "__main__":
    main()
