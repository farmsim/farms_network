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
  circle, minimum size=0.5 cm, inner sep=0.04cm, outer sep=0.04cm, draw=black, thick, double
}},
}}
{content}
\end{{document}}"""


def main():
    """Main."""

    # Main network
    network = nx.DiGraph()

    # Generate rhythm centers
    rhythm1 = RhythmGenerator.generate_nodes_edges(name="LEFT-FORE", anchor_x=-5.0, anchor_y=5.0)
    rhythm2 = RhythmGenerator.generate_nodes_edges(name="LEFT-HIND", anchor_x=-5.0, anchor_y=-5.0)
    rhythm3 = RhythmGenerator.generate_nodes_edges(name="RIGHT-FORE", anchor_x=5.0, anchor_y=5.0)
    rhythm4 = RhythmGenerator.generate_nodes_edges(name="RIGHT-HIND", anchor_x=5.0, anchor_y=-5.0)

    network = nx.compose_all(
        [rhythm1, rhythm2, rhythm3, rhythm4]
    )

    colors = {
        "RG-F": "fill=flex!40",
        "RG-E": "fill=ext!40",
        "In-E": "fill=inf!40",
        "In-F": "fill=inf!40",
    }

    node_options = {
        name: colors["-".join(name.split("-")[-2:])]
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


if __name__ == "__main__":
    main()
