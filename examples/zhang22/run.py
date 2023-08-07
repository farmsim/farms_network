""" Generate and reproduce Zhang, Shevtsova, et al. eLife 2022;11:e73424. DOI:
https://doi.org/10.7554/eLife.73424 paper network """


import networkx as nx
import numpy as np
import farms_pylog as pylog
import os


class RhythmGenerator(nx.DiGraph):
    """Generate RhythmGenerator Network"""

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
            self.name + "-RG-F",
            label="F",
            model="lif_danner_nap",
            x=1.0 + anchor_x,
            y=0.0 + anchor_y,
            color="r",
            m_e=0.1,
            b_e=0.0,
            v0=-62.5,
            h0=np.random.uniform(0, 1),
        )
        self.add_node(
            self.name + "-RG-E",
            label="E",
            model="lif_danner_nap",
            x=1.0 + anchor_x,
            y=4.0 + anchor_y,
            color="b",
            m_e=0.0,
            b_e=0.1,
            v0=-62.5,
            h0=np.random.uniform(0, 1),
        )
        self.add_node(
            self.name + "-In-F",
            label="In",
            model="lif_danner",
            x=0.0 + anchor_x,
            y=2.0 + anchor_y,
            color="m",
            v0=-60.0,
        )
        self.add_node(
            self.name + "-In-E",
            label="In",
            model="lif_danner",
            x=2.0 + anchor_x,
            y=2.0 + anchor_y,
            color="m",
            v0=-60.0,
        )

    def add_connections(self):
        self.add_edge(self.name + "-RG-F", self.name + "-In-F", weight=0.4)
        self.add_edge(self.name + "-In-F", self.name + "-RG-E", weight=-1.0)
        self.add_edge(self.name + "-RG-E", self.name + "-In-E", weight=0.4)
        self.add_edge(self.name + "-In-E", self.name + "-RG-F", weight=-0.08)


class Commissural(nx.DiGraph):
    """Commissural Network template."""


    def __init__(self, name="", anchor_x=0.0, anchor_y=0.0):
        """Initialization."""
        super().__init__(name=name)

    @classmethod
    def generate_nodes_edges(cls, name, anchor_x=0.0, anchor_y=0.0):
        obj = cls(name, anchor_x, anchor_y)
        obj.add_neurons(anchor_x, anchor_y)
        obj.add_connections()
        return obj

    def add_neurons(self, anchor_x, anchor_y,):
        """Add neurons."""

        # V3
        # for side in ("LEFT", "RIGHT"):
        #     for leg in ("FORE", "HIND"):
        self.add_node(
            self.name + "-V3-E-Left-Fore",
            label="V3-E",
            model="lif_danner",
            x=-1.0 + anchor_x,
            y=8.0 + anchor_y,
            color="g",
            v0=-60.0,
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
    network = nx.DiGraph()

    # Generate rhythm centers
    rhythm1 = RhythmGenerator.generate_nodes_edges(name="LEFT-FORE", anchor_x=-5.0, anchor_y=4.0)
    rhythm2 = RhythmGenerator.generate_nodes_edges(name="LEFT-HIND", anchor_x=-5.0, anchor_y=-6.0)
    rhythm3 = RhythmGenerator.generate_nodes_edges(name="RIGHT-FORE", anchor_x=5.0, anchor_y=4.0)
    rhythm4 = RhythmGenerator.generate_nodes_edges(name="RIGHT-HIND", anchor_x=5.0, anchor_y=-6.0)

    # Commissural
    commissural = Commissural.generate_nodes_edges(name="commissural")

    # LSPN
    lpsn = LPSN.generate_nodes_edges(name="lspn")

    network = nx.compose_all(
        [rhythm1, rhythm2, rhythm3, rhythm4, commissural, lpsn]
    )

    colors = {
        "RG-F": "fill=flex!40",
        "RG-E": "fill=ext!40",
        "In-E": "fill=inf!40",
        "In-F": "fill=inf!40",
        "V2a-diag": "fill=red!40",
        "commissural-CINi1": "fill=green!40",
        "commissural-V0V": "fill=red!40",
        "commissural-V0D": "fill=green!40",
        "commissural-V3": "fill=red!40",
        "commissural-V2a": "fill=green!40",
        "commissural-IniV0V": "fill=red!40",
    }

    node_options = {
        name: colors.get("-".join(name.split("-")[-2:]), "fill=yellow!40")
        for name, node in network.nodes.items()
    }

    nx.write_graphml(network, "./config/auto_zhang_2022.graphml")
    nx.write_latex(
        network,
        "zhang2022_figure.tex",
        pos={name: (node['x'], node['y']) for name, node in network.nodes.items()},
        as_document=True,
        caption="A path graph",
        latex_label="fig1",
        node_options=node_options,
        default_node_options="my-node",
        node_label={
            name: node["label"]
            for name, node in network.nodes.items()
        },
        default_edge_options="[color=black, thick, -{Latex[scale=1.0]}, bend left, looseness=0.75]",
        document_wrapper=_DOC_WRAPPER_TIKZ,
    )

    latex_code = nx.to_latex(network)  # a string rather than a file

    os.system("pdflatex zhang2022_figure.tex")


def main():
    """Main."""

    # Generate the network
    generate_network()

    # Run the network
    # run_network()


if __name__ == "__main__":
    main()
