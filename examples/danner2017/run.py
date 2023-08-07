#!/usr/bin/env python

""" Generate and reproduce Danner, et al. eLife 2017;
DOI: https://doi.org/10.7554/eLife.31050.004 paper network """

import networkx as nx
import numpy as np
import farms_pylog as pylog
import os


class RhythmGenerator(nx.DiGraph):
    """Generate Rhythm Network"""

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

    def add_neurons(self, anchor_x, anchor_y):
        """Add neurons."""
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
            label="$CIN_{i1}$",
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
            self.name + "-V3",
            label="$V3$",
            model="lif_danner",
            x=1.0 + anchor_x,
            y=5.0 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.add_node(
            self.name + "-V2a",
            label="$V2_a$",
            model="lif_danner",
            x=(-1.0 if self.name[-1] == "L" else 3.0) + anchor_x,
            y=1.0 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.add_node(
            self.name + "_IniV0V",
            label="$Ini_{V0_V}$",
            model="lif_danner",
            x=(-1.0 if self.name[-1] == "L" else 3.0) + anchor_x,
            y=3 + anchor_y,
            color="m",
            v0=-60.0,
        )

    def add_connections(self):
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
            label="Ini",
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
        ...


class ConnectRG2Commissural(object):
    """Connect a RG circuit with Commissural"""

    def __init__(self, rg_l, rg_r, comm_l, comm_r):
        """Initialization."""
        super(ConnectRG2Commissural, self).__init__()
        self.net = nx.compose_all([rg_l, rg_r, comm_l, comm_r])
        self.name = self.net.name[0] + "RG_COMM"

        # Methods
        self.connect_circuits()
        return

    def connect_circuits(self):
        """Connect Afferents's to Interneurons."""

        def _name(side, name):
            """Add the network name to the neuron."""
            return self.name[0] + side + "_" + name

        self.net.add_edge(_name("L", "RG_E"), _name("L", "CINi1"), weight=0.40)

        self.net.add_edge(_name("R", "RG_E"), _name("R", "CINi1"), weight=0.40)

        self.net.add_edge(_name("L", "RG_F"), _name("L", "V2a"), weight=1.0)

        self.net.add_edge(_name("R", "RG_F"), _name("R", "V2a"), weight=1.0)

        self.net.add_edge(_name("L", "RG_F"), _name("L", "V0D"), weight=0.70)

        self.net.add_edge(_name("R", "RG_F"), _name("R", "V0D"), weight=0.70)

        self.net.add_edge(_name("L", "RG_F"), _name("L", "V3"), weight=0.35)

        self.net.add_edge(_name("R", "RG_F"), _name("R", "V3"), weight=0.35)

        self.net.add_edge(_name("L", "RG_F"), _name("L", "V2a_diag"), weight=0.50)

        self.net.add_edge(_name("R", "RG_F"), _name("R", "V2a_diag"), weight=0.50)

        self.net.add_edge(_name("L", "CINi1"), _name("R", "RG_F"), weight=-0.03)

        self.net.add_edge(_name("R", "CINi1"), _name("L", "RG_F"), weight=-0.03)

        self.net.add_edge(_name("L", "V0V"), _name("R", "IniV0V"), weight=0.60)

        self.net.add_edge(_name("R", "V0V"), _name("L", "IniV0V"), weight=0.60)

        self.net.add_edge(_name("L", "V0D"), _name("R", "RG_F"), weight=-0.07)

        self.net.add_edge(_name("R", "V0D"), _name("L", "RG_F"), weight=-0.07)

        self.net.add_edge(_name("L", "V3"), _name("R", "RG_F"), weight=0.03)

        self.net.add_edge(_name("R", "V3"), _name("L", "RG_F"), weight=0.03)

        self.net.add_edge(_name("L", "V2a"), _name("L", "V0V"), weight=1.0)

        self.net.add_edge(_name("R", "V2a"), _name("R", "V0V"), weight=1.0)

        self.net.add_edge(_name("L", "IniV0V"), _name("L", "RG_F"), weight=-0.07)

        self.net.add_edge(_name("R", "IniV0V"), _name("R", "RG_F"), weight=-0.07)

        return self.net


class ConnectFore2Hind(object):
    """Connect a Fore limb circuit with Hind Limb"""

    def __init__(self, fore, hind, lspn_l, lspn_r):
        """Initialization."""
        super(ConnectFore2Hind, self).__init__()
        self.net = nx.compose_all([fore, hind, lspn_l, lspn_r])
        self.name = self.net.name[0] + "MODEL"

        # Methods
        self.connect_circuits()
        return

    def connect_circuits(self):
        """Connect CPG's to Interneurons."""

        def _name(side, name, f_h=""):
            """Add the network name to the neuron."""
            return f_h + side + "_" + name

        self.net.add_edge(
            _name("L", "RG_E", "F"), _name("L", "Sh2_hom_fh"), weight=0.50
        )

        self.net.add_edge(
            _name("R", "RG_E", "F"), _name("R", "Sh2_hom_fh"), weight=0.50
        )

        self.net.add_edge(
            _name("L", "RG_E", "H"), _name("L", "Sh2_hom_hf"), weight=0.50
        )

        self.net.add_edge(
            _name("R", "RG_E", "H"), _name("R", "Sh2_hom_hf"), weight=0.50
        )

        self.net.add_edge(
            _name("L", "RG_F", "F"), _name("L", "Ini_hom_fh"), weight=0.70
        )

        self.net.add_edge(
            _name("R", "RG_F", "F"), _name("R", "Ini_hom_fh"), weight=0.70
        )

        self.net.add_edge(_name("L", "RG_F", "F"), _name("L", "V0D_diag"), weight=0.50)

        self.net.add_edge(_name("R", "RG_F", "F"), _name("R", "V0D_diag"), weight=0.50)

        self.net.add_edge(
            _name("L", "V2a_diag", "F"), _name("L", "V0V_diag_fh"), weight=0.90
        )

        self.net.add_edge(
            _name("R", "V2a_diag", "F"), _name("R", "V0V_diag_fh"), weight=0.90
        )

        self.net.add_edge(
            _name("L", "V2a_diag", "H"), _name("L", "V0V_diag_hf"), weight=0.90
        )

        self.net.add_edge(
            _name("R", "V2a_diag", "H"), _name("R", "V0V_diag_hf"), weight=0.90
        )

        self.net.add_edge(
            _name("L", "V0D_diag"), _name("R", "RG_F", "H"), weight=-0.075
        )

        self.net.add_edge(
            _name("R", "V0D_diag"), _name("L", "RG_F", "H"), weight=-0.075
        )

        self.net.add_edge(
            _name("L", "V0V_diag_fh"), _name("R", "RG_F", "H"), weight=0.02
        )

        self.net.add_edge(
            _name("R", "V0V_diag_fh"), _name("L", "RG_F", "H"), weight=0.02
        )

        self.net.add_edge(
            _name("L", "V0V_diag_hf"), _name("R", "RG_F", "F"), weight=0.065
        )

        self.net.add_edge(
            _name("R", "V0V_diag_hf"), _name("L", "RG_F", "F"), weight=0.065
        )

        self.net.add_edge(
            _name("L", "Ini_hom_fh"), _name("L", "RG_F", "H"), weight=-0.01
        )

        self.net.add_edge(
            _name("R", "Ini_hom_fh"), _name("R", "RG_F", "H"), weight=-0.01
        )

        self.net.add_edge(
            _name("L", "Sh2_hom_fh"), _name("L", "RG_F", "H"), weight=0.01
        )

        self.net.add_edge(
            _name("R", "Sh2_hom_fh"), _name("R", "RG_F", "H"), weight=0.01
        )

        self.net.add_edge(
            _name("L", "Sh2_hom_hf"), _name("L", "RG_F", "F"), weight=0.125
        )

        self.net.add_edge(
            _name("R", "Sh2_hom_hf"), _name("R", "RG_F", "F"), weight=0.125
        )

        return self.net


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
    rhythm1 = RhythmGenerator.generate_nodes_edges(name="LEFT-FORE", anchor_x=0.0, anchor_y=10.0)
    rhythm2 = RhythmGenerator.generate_nodes_edges(name="LEFT-HIND", anchor_x=0.0, anchor_y=-5.0)
    rhythm3 = RhythmGenerator.generate_nodes_edges(name="RIGHT-FORE", anchor_x=10.0, anchor_y=10.0)
    rhythm4 = RhythmGenerator.generate_nodes_edges(name="RIGHT-HIND", anchor_x=10.0, anchor_y=-5.0)

    # Commissural
    comm1 = Commissural.generate_nodes_edges(
        'LEFT-FORE', anchor_x=2.5, anchor_y=9.0
    )
    comm2 = Commissural.generate_nodes_edges(
        'LEFT-HIND', anchor_x=2.5, anchor_y=-4.0
    )
    comm3 = Commissural.generate_nodes_edges(
        'RIGHT-FORE', anchor_x=7.0, anchor_y=9.0
    )
    comm4 = Commissural.generate_nodes_edges(
        'RIGHT-HIND', anchor_x=7.0, anchor_y=-4.0
    )

    # LSPN
    lpsn1 = LPSN.generate_nodes_edges(name="LEFT", anchor_x=-2.0, anchor_y=5.0)
    lpsn2 = LPSN.generate_nodes_edges(name="RIGHT", anchor_x=12.0, anchor_y=5.0)

    network = nx.compose_all(
        [
            rhythm1, rhythm2, rhythm3, rhythm4,
            comm1, comm2, comm3, comm4,
            lpsn1, lpsn2
        ]
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

    nx.write_graphml(network, "./config/auto_danner_2017.graphml")
    nx.write_latex(
        network,
        "danner2017_figure.tex",
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

    os.system("pdflatex danner2017_figure.tex")


def main():
    """Main."""

    # Generate the network
    generate_network()

    # Run the network
    # run_network()


if __name__ == "__main__":
    main()
