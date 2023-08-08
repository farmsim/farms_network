#!/usr/bin/env python

""" Generate and reproduce Danner, et al. eLife 2017;
DOI: https://doi.org/10.7554/eLife.31050.004 paper network """


import os

import farms_pylog as pylog
import networkx as nx
import numpy as np
from farms_container import Container
from farms_network.neural_system import NeuralSystem
from matplotlib import pyplot as plt


def multiply_transform(vec, transform_mat) -> list:
    """Multiply 2D veector with 2D transformation matrix (3x3)"""
    pos = (transform_mat @ np.array(vec)).tolist()
    return pos


class RhythmGenerator(nx.DiGraph):
    """Generate Rhythm Network"""

    def __init__(self, name=""):
        """Initialization."""
        super().__init__(name=name)

    @classmethod
    def generate_nodes_edges(cls, name, transform_mat=np.identity(2)):
        obj = cls(name)
        obj.add_neurons(transform_mat)
        obj.add_connections()
        return obj

    def add_neurons(self, transform_mat):
        """Add neurons."""
        pos = multiply_transform((0.0, -4.0, 1.0), transform_mat)
        self.add_node(
            self.name + "-RG-F",
            label="F",
            model="lif_danner_nap",
            x=pos[0],
            y=pos[1],
            color="r",
            m_e=0.1,
            b_e=0.0,
            v0=-62.5,
            h0=np.random.uniform(0, 1),
        )
        pos = multiply_transform((0.0, 4.0, 1.0), transform_mat)
        self.add_node(
            self.name + "-RG-E",
            label="E",
            model="lif_danner_nap",
            x=pos[0],
            y=pos[1],
            color="b",
            m_e=0.0,
            b_e=0.1,
            v0=-62.5,
            h0=np.random.uniform(0, 1),
        )
        pos = multiply_transform((-2.0, 0.0, 1.0), transform_mat)
        self.add_node(
            self.name + "-In-F",
            label="In",
            model="lif_danner",
            x=pos[0],
            y=pos[1],
            color="m",
            v0=-60.0,
        )
        pos = multiply_transform((2.0, 0.0, 1.0), transform_mat)
        self.add_node(
            self.name + "-In-E",
            label="In",
            model="lif_danner",
            x=pos[0],
            y=pos[1],
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

    def __init__(self, name=""):
        """Initialization."""
        super().__init__(name=name)

    @classmethod
    def generate_nodes_edges(cls, name, transform_mat=np.identity(2)):
        obj = cls(name)
        obj.add_neurons(transform_mat)
        obj.add_connections()
        return obj

    def add_neurons(self, transform_mat):
        """Add neurons."""
        pos = multiply_transform(
            ((-1.0 if self.name[-1] == "L" else 3.0), 8.0, 1.0), transform_mat
        )
        self.add_node(
            self.name + "-V2a-diag",
            label="$V2_a$",
            x=pos[0],
            y=pos[1],
            model="lif_danner",
            color="g",
            v0=-60.0,
        )
        pos = multiply_transform((1.0, -1.0, 1.0), transform_mat)
        self.add_node(
            self.name + "-CINi1",
            label="$IN_{i1}$",
            x=pos[0],
            y=pos[1],
            model="lif_danner",
            color="m",
            v0=-60.0,
        )
        pos = multiply_transform((1.0, 1.0, 1.0), transform_mat)
        self.add_node(
            self.name + "-V0V",
            label="$V0_V$",
            x=pos[0],
            y=pos[1],
            model="lif_danner",
            color="g",
            m_i=0.15,
            b_i=0.0,
            v0=-60.0,
        )
        pos = multiply_transform((1.0, 3.0, 1.0), transform_mat)
        self.add_node(
            self.name + "-V0D",
            label="$V0_D$",
            x=pos[0],
            y=pos[1],
            model="lif_danner",
            color="m",
            m_i=0.75,
            b_i=0.0,
            v0=-60.0,
        )
        pos = multiply_transform((1.0, 5.0, 1.0), transform_mat)
        self.add_node(
            self.name + "-V3",
            label="$V3$",
            x=pos[0],
            y=pos[1],
            model="lif_danner",
            color="g",
            v0=-60.0,
        )
        pos = multiply_transform(
            ((-1.0 if self.name[-1] == "L" else 3.0), 1.0, 1.0), transform_mat
        )
        self.add_node(
            self.name + "-V2a",
            label="$V2_a$",
            x=pos[0],
            y=pos[1],
            model="lif_danner",
            color="g",
            v0=-60.0,
        )
        pos = multiply_transform(
            ((-1.0 if self.name[-1] == "L" else 3.0), 3.0, 1.0), transform_mat
        )
        self.add_node(
            self.name + "-IniV0V",
            label="$V0_V$",
            x=pos[0],
            y=pos[1],
            model="lif_danner",
            color="m",
            v0=-60.0,
        )

    def add_connections(self):
        ...


class LPSN(nx.DiGraph):
    """Long Propriospinal Neuron Network template."""

    def __init__(self, name=""):
        """Initialization."""
        super().__init__(name=name)

    @classmethod
    def generate_nodes_edges(cls, name, transform_mat=np.identity(2)):
        obj = cls(name)
        obj.add_neurons(transform_mat)
        obj.add_connections()
        return obj

    def add_neurons(self, transform_mat):
        """Add neurons."""
        pos = multiply_transform((-5.0, 0.0, 1.0), transform_mat)
        self.add_node(
            self.name + "-V0D-diag",
            label="$V0_D$",
            model="lif_danner",
            x=pos[0],
            y=pos[1],
            color="m",
            m_i=0.75,
            b_i=0.0,
            v0=-60.0,
        )
        pos = multiply_transform((-5.0, 2.0, 1.0), transform_mat)
        self.add_node(
            self.name + "-V0V-diag-fh",
            label="$V0_V$",
            model="lif_danner",
            x=pos[0],
            y=pos[1],
            color="g",
            v0=-60.0,
        )
        pos = multiply_transform((-5.0, 4.0, 1.0), transform_mat)
        self.add_node(
            self.name + "-V0V-diag-hf",
            label="$V0_V$",
            model="lif_danner",
            x=pos[0],
            y=pos[1],
            color="g",
            v0=-60.0,
        )
        pos = multiply_transform((-1.0, 4.0, 1.0), transform_mat)
        self.add_node(
            self.name + "-Ini-hom-fh",
            label="Ini",
            model="lif_danner",
            x=pos[0],
            y=pos[1],
            color="g",
            v0=-60.0,
        )
        pos = multiply_transform(
            ((-4.0 if self.name[-1] == "L" else 6.0), 4.0, 1.0), transform_mat
        )
        self.add_node(
            self.name + "-Sh2-hom-fh",
            label="$Sh_2$",
            model="lif_danner",
            x=pos[0],
            y=pos[1],
            color="g",
            v0=-60.0,
        )
        pos = multiply_transform(
            ((-4.0 if self.name[-1] == "L" else 6.0), 0.0, 1.0), transform_mat
        )
        self.add_node(
            self.name + "-Sh2-hom-hf",
            label="$Sh_2$",
            model="lif_danner",
            x=pos[0],
            y=pos[1],
            color="g",
            v0=-60.0,
        )

    def add_connections(self):
        ...


def connect_rg_commissural(network):
    """Connect RG's to Commissural."""

    def _name(ipsi, contra, name):
        """Add the network name to the neuron."""
        return ipsi + "-" + contra + "-" + name

    for contra in ("HIND", "FORE"):

        network.add_edge(
            _name("LEFT", contra, "RG-E"), _name("LEFT", contra, "CINi1"), weight=0.40
        )

        network.add_edge(
            _name("RIGHT", contra, "RG-E"), _name("RIGHT", contra, "CINi1"), weight=0.40
        )

        network.add_edge(
            _name("LEFT", contra, "RG-F"), _name("LEFT", contra, "V2a"), weight=1.0
        )

        network.add_edge(
            _name("RIGHT", contra, "RG-F"), _name("RIGHT", contra, "V2a"), weight=1.0
        )

        network.add_edge(
            _name("LEFT", contra, "RG-F"), _name("LEFT", contra, "V0D"), weight=0.70
        )

        network.add_edge(
            _name("RIGHT", contra, "RG-F"), _name("RIGHT", contra, "V0D"), weight=0.70
        )

        network.add_edge(
            _name("LEFT", contra, "RG-F"), _name("LEFT", contra, "V3"), weight=0.35
        )

        network.add_edge(
            _name("RIGHT", contra, "RG-F"), _name("RIGHT", contra, "V3"), weight=0.35
        )

        network.add_edge(
            _name("LEFT", contra, "RG-F"),
            _name("LEFT", contra, "V2a-diag"),
            weight=0.50,
        )

        network.add_edge(
            _name("RIGHT", contra, "RG-F"),
            _name("RIGHT", contra, "V2a-diag"),
            weight=0.50,
        )

        network.add_edge(
            _name("LEFT", contra, "CINi1"), _name("RIGHT", contra, "RG-F"), weight=-0.03
        )

        network.add_edge(
            _name("RIGHT", contra, "CINi1"), _name("LEFT", contra, "RG-F"), weight=-0.03
        )

        network.add_edge(
            _name("LEFT", contra, "V0V"), _name("RIGHT", contra, "IniV0V"), weight=0.60
        )

        network.add_edge(
            _name("RIGHT", contra, "V0V"), _name("LEFT", contra, "IniV0V"), weight=0.60
        )

        network.add_edge(
            _name("LEFT", contra, "V0D"), _name("RIGHT", contra, "RG-F"), weight=-0.07
        )

        network.add_edge(
            _name("RIGHT", contra, "V0D"), _name("LEFT", contra, "RG-F"), weight=-0.07
        )

        network.add_edge(
            _name("LEFT", contra, "V3"), _name("RIGHT", contra, "RG-F"), weight=0.03
        )

        network.add_edge(
            _name("RIGHT", contra, "V3"), _name("LEFT", contra, "RG-F"), weight=0.03
        )

        network.add_edge(
            _name("LEFT", contra, "V2a"), _name("LEFT", contra, "V0V"), weight=1.0
        )

        network.add_edge(
            _name("RIGHT", contra, "V2a"), _name("RIGHT", contra, "V0V"), weight=1.0
        )

        network.add_edge(
            _name("LEFT", contra, "IniV0V"), _name("LEFT", contra, "RG-F"), weight=-0.07
        )

        network.add_edge(
            _name("RIGHT", contra, "IniV0V"),
            _name("RIGHT", contra, "RG-F"),
            weight=-0.07,
        )


def connect_fore_hind_circuits(network):
    """Connect CPG's to Interneurons."""

    def _name(side, name, f_h=""):
        """Add the network name to the neuron."""
        if f_h:
            return side + "-" + f_h + "-" + name
        return side + "-" + name

    network.add_edge(
        _name("LEFT", "RG-E", "FORE"), _name("LEFT", "Sh2-hom-fh"), weight=0.50
    )

    network.add_edge(
        _name("RIGHT", "RG-E", "FORE"), _name("RIGHT", "Sh2-hom-fh"), weight=0.50
    )

    network.add_edge(
        _name("LEFT", "RG-E", "HIND"), _name("LEFT", "Sh2-hom-hf"), weight=0.50
    )

    network.add_edge(
        _name("RIGHT", "RG-E", "HIND"), _name("RIGHT", "Sh2-hom-hf"), weight=0.50
    )

    network.add_edge(
        _name("LEFT", "RG-F", "FORE"), _name("LEFT", "Ini-hom-fh"), weight=0.70
    )

    network.add_edge(
        _name("RIGHT", "RG-F", "FORE"), _name("RIGHT", "Ini-hom-fh"), weight=0.70
    )

    network.add_edge(
        _name("LEFT", "RG-F", "FORE"), _name("LEFT", "V0D-diag"), weight=0.50
    )

    network.add_edge(
        _name("RIGHT", "RG-F", "FORE"), _name("RIGHT", "V0D-diag"), weight=0.50
    )

    network.add_edge(
        _name("LEFT", "V2a-diag", "FORE"), _name("LEFT", "V0V-diag-fh"), weight=0.90
    )

    network.add_edge(
        _name("RIGHT", "V2a-diag", "FORE"), _name("RIGHT", "V0V-diag-fh"), weight=0.90
    )

    network.add_edge(
        _name("LEFT", "V2a-diag", "HIND"), _name("LEFT", "V0V-diag-hf"), weight=0.90
    )

    network.add_edge(
        _name("RIGHT", "V2a-diag", "HIND"), _name("RIGHT", "V0V-diag-hf"), weight=0.90
    )

    network.add_edge(
        _name("LEFT", "V0D-diag"), _name("RIGHT", "RG-F", "HIND"), weight=-0.075
    )

    network.add_edge(
        _name("RIGHT", "V0D-diag"), _name("LEFT", "RG-F", "HIND"), weight=-0.075
    )

    network.add_edge(
        _name("LEFT", "V0V-diag-fh"), _name("RIGHT", "RG-F", "HIND"), weight=0.02
    )

    network.add_edge(
        _name("RIGHT", "V0V-diag-fh"), _name("LEFT", "RG-F", "HIND"), weight=0.02
    )

    network.add_edge(
        _name("LEFT", "V0V-diag-hf"), _name("RIGHT", "RG-F", "FORE"), weight=0.065
    )

    network.add_edge(
        _name("RIGHT", "V0V-diag-hf"), _name("LEFT", "RG-F", "FORE"), weight=0.065
    )

    network.add_edge(
        _name("LEFT", "Ini-hom-fh"), _name("LEFT", "RG-F", "HIND"), weight=-0.01
    )

    network.add_edge(
        _name("RIGHT", "Ini-hom-fh"), _name("RIGHT", "RG-F", "HIND"), weight=-0.01
    )

    network.add_edge(
        _name("LEFT", "Sh2-hom-fh"), _name("LEFT", "RG-F", "HIND"), weight=0.01
    )

    network.add_edge(
        _name("RIGHT", "Sh2-hom-fh"), _name("RIGHT", "RG-F", "HIND"), weight=0.01
    )

    network.add_edge(
        _name("LEFT", "Sh2-hom-hf"), _name("LEFT", "RG-F", "FORE"), weight=0.125
    )

    network.add_edge(
        _name("RIGHT", "Sh2-hom-hf"), _name("RIGHT", "RG-F", "FORE"), weight=0.125
    )


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
\definecolor{{flex}}{{RGB}}{{255,0,32}}
\definecolor{{inf}}{{RGB}}{{143,37,87}}
\definecolor{{ext}}{{RGB}}{{0,90,172}}

\tikzset{{
% Color shades
my-node/.style={{
  circle, minimum size=0.75 cm, inner sep=0.01cm, outer sep=0.01cm, draw=black, thick, double, font={{\footnotesize}}
}},
}}
{content}
\end{{document}}"""

_FIG_WRAPPER = r"""\begin{{figure}}
\centering
{content}{caption}{label}
\end{{figure}}"""


def generate_network():
    """Generate network"""

    # Main network
    network = nx.DiGraph()

    # Generate rhythm centers
    scale = 0.6
    scale_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    angle = np.radians(60)
    off_x, off_y = -5.0, 5.0
    rhythm1 = RhythmGenerator.generate_nodes_edges(
        name="LEFT-FORE",
        transform_mat=np.array(
            [
                [np.cos(angle), -np.sin(angle), off_x],
                [np.sin(angle), np.cos(angle), off_y],
                [0, 0, 1],
            ]
        )
        @ scale_matrix,
    )
    angle = np.radians(120)
    off_x, off_y = -5.0, -5.0
    rhythm2 = RhythmGenerator.generate_nodes_edges(
        name="LEFT-HIND",
        transform_mat=np.array(
            [
                [np.cos(angle), -np.sin(angle), off_x],
                [np.sin(angle), np.cos(angle), off_y],
                [0, 0, 1],
            ]
        )
        @ scale_matrix,
    )
    angle = np.radians(-60)
    off_x, off_y = 5.0, 5.0
    rhythm3 = RhythmGenerator.generate_nodes_edges(
        name="RIGHT-FORE",
        transform_mat=np.array(
            [
                [np.cos(angle), -np.sin(angle), off_x],
                [np.sin(angle), np.cos(angle), off_y],
                [0, 0, 1],
            ]
        )
        @ scale_matrix,
    )
    angle = np.radians(-120)
    off_x, off_y = 5.0, -5.0
    rhythm4 = RhythmGenerator.generate_nodes_edges(
        name="RIGHT-HIND",
        transform_mat=np.array(
            [
                [np.cos(angle), -np.sin(angle), off_x],
                [np.sin(angle), np.cos(angle), off_y],
                [0, 0, 1],
            ]
        )
        @ scale_matrix,
    )

    # Commissural
    off_x, off_y = 0.0, -12.5
    flip_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    pos_matrix = np.array([[1, 0, off_x], [0, 1, off_y], [0, 0, 1]])
    comm1 = Commissural.generate_nodes_edges(
        "LEFT-FORE", transform_mat=scale_matrix @ flip_matrix @ pos_matrix
    )
    off_x, off_y = 0.0, -12.5
    flip_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    pos_matrix = np.array([[1, 0, off_x], [0, 1, off_y], [0, 0, 1]])
    comm3 = Commissural.generate_nodes_edges(
        "LEFT-HIND", transform_mat=scale_matrix @ flip_matrix @ pos_matrix
    )
    off_x, off_y = 0.0, -12.5
    flip_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    pos_matrix = np.array([[1, 0, off_x], [0, 1, off_y], [0, 0, 1]])
    comm2 = Commissural.generate_nodes_edges(
        "RIGHT-FORE", transform_mat=scale_matrix @ flip_matrix @ pos_matrix
    )
    off_x, off_y = 0.0, -12.5
    flip_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    pos_matrix = np.array([[1, 0, off_x], [0, 1, off_y], [0, 0, 1]])
    comm4 = Commissural.generate_nodes_edges(
        "RIGHT-HIND", transform_mat=scale_matrix @ flip_matrix @ pos_matrix
    )

    # LSPN
    off_x, off_y = 6.0, -2.0
    flip_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    pos_matrix = np.array([[1, 0, off_x], [0, 1, off_y], [0, 0, 1]])
    lpsn1 = LPSN.generate_nodes_edges(
        name="LEFT", transform_mat=scale_matrix @ flip_matrix @ pos_matrix
    )
    off_x, off_y = 6.0, -2.0
    flip_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    pos_matrix = np.array([[1, 0, off_x], [0, 1, off_y], [0, 0, 1]])
    lpsn2 = LPSN.generate_nodes_edges(
        name="RIGHT", transform_mat=scale_matrix @ flip_matrix @ pos_matrix
    )

    network = nx.compose_all(
        [rhythm1, rhythm2, rhythm3, rhythm4, comm1, comm2, comm3, comm4, lpsn1, lpsn2]
    )

    # Connections
    connect_rg_commissural(network)

    connect_fore_hind_circuits(network)

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
    nx.write_graphml(network, "./config/auto_danner_2017.graphml")
    # for name, node in network.nodes.items():
    #     print(name)
    #     print(node["x"])
    nx.write_latex(
        network,
        "danner2017_figure.tex",
        pos={name: (node["x"], node["y"]) for name, node in network.nodes.items()},
        as_document=True,
        caption="A path graph",
        latex_label="fig1",
        node_options=node_options,
        default_node_options="my-node",
        node_label={name: node["label"] for name, node in network.nodes.items()},
        default_edge_options="[color=black, ultra thick, -{Latex[scale=1.0]}, on background layer]",
        document_wrapper=_DOC_WRAPPER_TIKZ,
        figure_wrapper=_FIG_WRAPPER,
    )

    latex_code = nx.to_latex(network)  # a string rather than a file

    os.system("pdflatex danner2017_figure.tex")

    container = Container()

    # # Initialize network
    net_ = NeuralSystem("./config/auto_danner_2017.graphml", container)

    edge_colors = [
        network.nodes[name[0]]['color']
        for name, edge in network.edges.items() 
    ]

    fig = net_.visualize_network(
        node_size=1500,
        node_labels=True,
        alpha=0.5,
        edge_labels=False,
        edge_alpha=False,
        color_map_edge=edge_colors,
        plt_out=plt
    )
    plt.show()


def main():
    """Main."""

    # Generate the network
    generate_network()

    # Run the network
    # run_network()


if __name__ == "__main__":
    main()
