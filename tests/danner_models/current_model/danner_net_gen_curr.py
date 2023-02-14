#!/usr/bin/env python

""" Generate Danner Network. Current Model"""

import networkx as nx
import numpy as np
import farms_pylog as pylog


class CPG(object):
    """Generate CPG Network"""

    def __init__(self, name, anchor_x=0.0, anchor_y=0.0):
        """Initialization."""
        super(CPG, self).__init__()
        self.cpg = nx.DiGraph()
        self.name = name
        self.cpg.name = name

        # Methods
        self.add_neurons(anchor_x, anchor_y)
        self.add_connections()
        return

    def add_neurons(self, anchor_x, anchor_y):
        """Add neurons."""
        self.cpg.add_node(
            self.name + "_RG_F",
            model="lif_danner_nap",
            x=1.0 + anchor_x,
            y=0.0 + anchor_y,
            color="r",
            m_e=0.1,
            b_e=0.0,
            v0=-62.5,
            h0=np.random.uniform(0, 1),
        )
        self.cpg.add_node(
            self.name + "_RG_E",
            model="lif_danner_nap",
            x=1.0 + anchor_x,
            y=4.0 + anchor_y,
            color="b",
            m_e=0.0,
            b_e=0.1,
            v0=-62.5,
            h0=np.random.uniform(0, 1),
        )
        self.cpg.add_node(
            self.name + "_In_F",
            model="lif_danner",
            x=0.0 + anchor_x,
            y=2.0 + anchor_y,
            color="m",
            v0=-60.0,
        )
        self.cpg.add_node(
            self.name + "_In_E",
            model="lif_danner",
            x=2.0 + anchor_x,
            y=2.0 + anchor_y,
            color="m",
            v0=-60.0,
        )

    def add_connections(self):
        self.cpg.add_edge(self.name + "_RG_F", self.name + "_In_F", weight=0.4)
        self.cpg.add_edge(self.name + "_In_F", self.name + "_RG_E", weight=-1.0)
        self.cpg.add_edge(self.name + "_RG_E", self.name + "_In_E", weight=0.4)
        self.cpg.add_edge(self.name + "_In_E", self.name + "_RG_F", weight=-0.08)
        return


class Commissural(object):
    """Commissural Network template."""

    def __init__(self, name, anchor_x=0.0, anchor_y=0.0, color="y"):
        """Initialization."""
        super(Commissural, self).__init__()
        self.commissural = nx.DiGraph()
        self.name = name
        self.commissural.name = name

        # Methods
        self.add_neurons(anchor_x, anchor_y, color)
        self.add_connections()
        return

    def add_neurons(self, anchor_x, anchor_y, color):
        """Add neurons."""
        self.commissural.add_node(
            self.name + "_V2a_diag",
            x=(-1.0 if self.name[-1] == "L" else 3.0) + anchor_x,
            model="lif_danner",
            y=6.0 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.commissural.add_node(
            self.name + "_CINi1",
            model="lif_danner",
            x=1.0 + anchor_x,
            y=-1.0 + anchor_y,
            color="m",
            v0=-60.0,
        )
        self.commissural.add_node(
            self.name + "_V0V",
            model="lif_danner",
            x=1.0 + anchor_x,
            y=1.0 + anchor_y,
            color="g",
            m_i=0.15,
            b_i=0.0,
            v0=-60.0,
        )
        self.commissural.add_node(
            self.name + "_V0D",
            model="lif_danner",
            x=1.0 + anchor_x,
            y=3.0 + anchor_y,
            color="m",
            m_i=0.75,
            b_i=0.0,
            v0=-60.0,
        )
        self.commissural.add_node(
            self.name + "_V3",
            model="lif_danner",
            x=1.0 + anchor_x,
            y=5.0 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.commissural.add_node(
            self.name + "_V2a",
            model="lif_danner",
            x=(-1.0 if self.name[-1] == "L" else 3.0) + anchor_x,
            y=1.0 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.commissural.add_node(
            self.name + "_IniV0V",
            model="lif_danner",
            x=(-1.0 if self.name[-1] == "L" else 3.0) + anchor_x,
            y=3 + anchor_y,
            color=color,
            v0=-60.0,
        )
        return

    def add_connections(self):
        return


class LPSN(object):
    """Long Propriospinal Neuron Network template."""

    def __init__(self, name, anchor_x=0.0, anchor_y=0.0, color="c"):
        """Initialization."""
        super(LPSN, self).__init__()
        self.lpsn = nx.DiGraph()
        self.name = name
        self.lpsn.name = name

        # Methods
        self.add_neurons(anchor_x, anchor_y, color)
        self.add_connections()
        return

    def add_neurons(self, anchor_x, anchor_y, color):
        """Add neurons."""
        self.lpsn.add_node(
            self.name + "_V0D_diag",
            model="lif_danner",
            x=1.0 + anchor_x,
            y=0.0 + anchor_y,
            color="m",
            m_i=0.75,
            b_i=0.0,
            v0=-60.0,
        )
        self.lpsn.add_node(
            self.name + "_V0V_diag_fh",
            model="lif_danner",
            x=1.0 + anchor_x,
            y=2 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.lpsn.add_node(
            self.name + "_V0V_diag_hf",
            model="lif_danner",
            x=1.0 + anchor_x,
            y=4 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.lpsn.add_node(
            self.name + "_Ini_hom_fh",
            model="lif_danner",
            x=(-2.0 if self.name[-1] == "L" else 4.0) + anchor_x,
            y=4 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.lpsn.add_node(
            self.name + "_Sh2_hom_fh",
            model="lif_danner",
            x=(-4.0 if self.name[-1] == "L" else 6.0) + anchor_x,
            y=0 + anchor_y,
            color="g",
            v0=-60.0,
        )
        self.lpsn.add_node(
            self.name + "_Sh2_hom_hf",
            model="lif_danner",
            x=(-4.0 if self.name[-1] == "L" else 6.0) + anchor_x,
            y=4 + anchor_y,
            color="g",
            v0=-60.0,
        )
        return

    def add_connections(self):
        return


class Motorneurons(object):
    """Motorneurons layers. Also contains interneurons."""

    def __init__(
        self,
        name,
        muscles=None,
        antagonists=None,
        agonists=None,
        anchor_x=0.0,
        anchor_y=0.0,
        color=["k", "m", "m"],
    ):
        super(Motorneurons, self).__init__()
        self.name = name
        self.net = nx.DiGraph()

        # Methods
        antagonists = [] if not antagonists else antagonists
        agonists = [] if not agonists else agonists
        self.add_neurons(muscles, anchor_x, anchor_y, color)
        self.add_connections(muscles, antagonists, agonists)

        return

    def add_neurons(self, muscles, anchor_x, anchor_y, color):
        """Add neurons."""
        pylog.debug("Adding motorneurons")
        _num_muscles = np.size(muscles)
        _pos = np.arange(-_num_muscles, _num_muscles, 2.0)

        for j, muscle in enumerate(muscles):
            self.net.add_node(
                self.name + "_Mn_" + muscle,
                model="lif_danner",
                x=float(_pos[j]) + anchor_x,
                y=0.0 + anchor_y,
                color=color[0],
                v0=-60.0,
                e_leak=-52.5,
                g_leak=1.0,
            )

            self.net.add_node(
                self.name + "_Rn_" + muscle,
                model="lif_danner",
                x=float(_pos[j]) + anchor_x - 1.0,
                y=2.0 + anchor_y,
                color=color[1],
                v0=-60.0,
            )

            self.net.add_node(
                self.name + "_IaIn_" + muscle,
                model="lif_danner",
                x=float(_pos[j]) + anchor_x + 1.0,
                y=2.0 + anchor_y,
                color=color[2],
                v0=-60.0,
            )

            self.net.add_node(
                self.name + "_IbIn_" + muscle,
                model="lif_danner",
                x=float(_pos[j]) + anchor_x + 1.0,
                y=2.0 + anchor_y,
                color=color[2],
                v0=-60.0,
            )

    def add_connections(self, muscles, antagonists, agonists):
        """Connect the neurons."""
        for muscle in muscles:
            self.net.add_edge(
                self.name + "_Mn_" + muscle, self.name + "_Rn_" + muscle, weight=1.0
            )
            self.net.add_edge(
                self.name + "_Mn_" + muscle, self.name + "_IaIn_" + muscle, weight=1.0
            )
            self.net.add_edge(
                self.name + "_Rn_" + muscle, self.name + "_Mn_" + muscle, weight=-1.0
            )
            self.net.add_edge(
                self.name + "_IbIn_" + muscle, self.name + "_Mn_" + muscle, weight=1.0
            )

            if muscle in antagonists:
                for antagonist in antagonists:
                    self.net.add_edge(
                        self.name + "_IaIn_" + muscle,
                        self.name + "_Mn_" + antagonist,
                        weight=-1.0,
                    )
                    self.net.add_edge(
                        self.name + "_Rn_" + muscle,
                        self.name + "_IaIn_" + antagonist,
                        weight=-1.0,
                    )

            if muscle in agonists:
                for agonist in agonists:
                    self.net.add_edge(
                        self.name + "_Rn_" + muscle,
                        self.name + "_Mn_" + agonist,
                        weight=-1.0,
                    )
        return self.net


class Afferents(object):
    """Generate Afferents Network"""

    def __init__(self, name, muscles, anchor_x=0.0, anchor_y=0.0, color="y"):
        """Initialization."""
        super(Afferents, self).__init__()
        self.afferents = nx.DiGraph()
        self.name = name
        self.afferents.name = name

        # Methods
        self.add_neurons(muscles, anchor_x, anchor_y, color)
        return

    def add_neurons(self, muscles, anchor_x, anchor_y, color):
        """Add neurons."""
        pylog.debug("Adding sensory afferents")
        _num_muscles = np.size(muscles)
        _pos = np.arange(-_num_muscles, _num_muscles, 2.0)

        self.afferents.add_node(
            self.name + "_Plantar_Cutaneous",
            model="sensory",
            x=float(_pos[int(np.floor(_num_muscles / 2.0))]) + anchor_x,
            y=5.0 + anchor_y,
            color=color,
            init=0.0,
        )

        for j, muscle in enumerate(muscles):
            self.afferents.add_node(
                self.name + "_" + muscle + "_Ia",
                model="sensory",
                x=float(_pos[j]) + anchor_x,
                y=0.0 + anchor_y,
                color=color,
                init=0.0,
            )

            self.afferents.add_node(
                self.name + "_" + muscle + "_II",
                model="sensory",
                x=float(_pos[j]) + anchor_x,
                y=3.0 + anchor_y,
                color=color,
                init=0.0,
            )

            self.afferents.add_node(
                self.name + "_" + muscle + "_Ib",
                model="sensory",
                x=float(_pos[j]) + anchor_x,
                y=-3.0 + anchor_y,
                color=color,
                init=0.0,
            )

    def add_connections(self):

        return


class ConnectAfferents2CPG(object):
    """Connect a PF circuit with RG"""

    def __init__(self, cpg, afferents, muscles, flex_RGF_muscles, ext_RGE_muscles):
        """Initialization."""
        super(ConnectAfferents2CPG, self).__init__()
        self.net = nx.compose_all([cpg, afferents])
        self.name = self.net.name
        # Methods
        self.connect_circuits(muscles, flex_RGF_muscles, ext_RGE_muscles)
        return

    def connect_circuits(self, muscles, flex_RGF_muscles, ext_RGE_muscles):
        """Connect CPG's to Interneurons."""

        self.net.add_edge(
            self.name + "_Plantar_Cutaneous", self.name + "_RG_E", weight=1.0
        )
        self.net.add_edge(
            self.name + "_Plantar_Cutaneous", self.name + "_In_E", weight=1.0
        )
        for muscle in flex_RGF_muscles:
            if muscle in muscles:
                self.net.add_edge(
                    self.name + "_" + muscle + "_II", self.name + "_RG_F", weight=1.0
                )
                self.net.add_edge(
                    self.name + "_" + muscle + "_II", self.name + "_In_F", weight=1.0
                )

        for muscle in ext_RGE_muscles:
            if muscle in muscles:
                self.net.add_edge(
                    self.name + "_" + muscle + "_Ib", self.name + "_RG_E", weight=1.0
                )
                self.net.add_edge(
                    self.name + "_" + muscle + "_Ib", self.name + "_In_E", weight=1.0
                )

        for muscle in muscles:
            self.net.add_edge(
                self.name + "_" + muscle + "_Ia",
                self.name + "_Mn_" + muscle,
                weight=1.0,
            )
            self.net.add_edge(
                self.name + "_" + muscle + "_Ib",
                self.name + "_Mn_" + muscle,
                weight=1.0,
            )
            self.net.add_edge(
                self.name + "_" + muscle + "_Ia",
                self.name + "_IaIn_" + muscle,
                weight=1.0,
            )
            self.net.add_edge(
                self.name + "_" + muscle + "_Ib",
                self.name + "_IbIn_" + muscle,
                weight=1.0,
            )
        return self.net


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


class PatternFormation(object):
    """Pattern Formation Layer"""

    def __init__(self, name, anchor_x=0.0, anchor_y=0.0, color="g"):
        super(PatternFormation, self).__init__()
        self.name = name
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y
        self.color = color

        self.pf_net = nx.DiGraph()

        # Methods
        self.add_neurons(anchor_x, anchor_y, color)
        self.add_connections()

        return

    def add_neurons(self, anchor_x, anchor_y, color):
        """Add neurons."""
        self.pf_net.add_node(
            self.name + "_PF_F",
            model="lif_danner_nap",
            x=-3.0 + anchor_x,
            y=0.0 + anchor_y,
            color="r",
            m_e=0.6,
            b_e=0.0,
            v0=-60.0,
            g_nap=0.125,
            e_leak=-67.5,
            h0=np.random.uniform(0, 1),
        )

        self.pf_net.add_node(
            self.name + "_PF_E",
            model="lif_danner_nap",
            x=-3.0 + anchor_x,
            y=4.0 + anchor_y,
            color="b",
            m_e=0.60,
            b_e=0.0,
            v0=-60.0,
            g_nap=0.125,
            e_leak=-67.5,
            h0=np.random.uniform(0, 1),
        )

        self.pf_net.add_node(
            self.name + "_Inp_F",
            model="lif_danner",
            x=-4.0 + anchor_x,
            y=2.0 + anchor_y,
            color="m",
            v0=-60.0,
        )

        self.pf_net.add_node(
            self.name + "_Inp_E",
            model="lif_danner",
            x=-2.0 + anchor_x,
            y=2.0 + anchor_y,
            color="m",
            v0=-60.0,
        )

        self.pf_net.add_node(
            self.name + "_PF_Sw",
            model="lif_danner_nap",
            x=5.0 + anchor_x,
            y=0.0 + anchor_y,
            color="r",
            v0=-60.0,
            g_nap=0.125,
            e_leak=-67.5,
            g_leak=1.0,
            h0=np.random.uniform(0, 1),
        )

        self.pf_net.add_node(
            self.name + "_PF_St",
            model="lif_danner_nap",
            x=5.0 + anchor_x,
            y=4.0 + anchor_y,
            color="b",
            m_e=0.07,
            b_e=0.0,
            v0=-60.0,
            g_nap=0.125,
            e_leak=-67.5,
            g_leak=1.0,
            h0=np.random.uniform(0, 1),
        )

        self.pf_net.add_node(
            self.name + "_Inp_Sw",
            model="lif_danner",
            x=6.0 + anchor_x,
            y=2.0 + anchor_y,
            color="m",
            v0=-60.0,
        )

        self.pf_net.add_node(
            self.name + "_Inp_St",
            model="lif_danner",
            x=4.0 + anchor_x,
            y=2.0 + anchor_y,
            color="m",
            v0=-60.0,
        )

        self.pf_net.add_node(
            self.name + "_Inp_F_Sw",
            model="lif_danner",
            x=6.0 + anchor_x,
            y=0.0 + anchor_y,
            color="m",
            v0=-60.0,
            g_leak=5.0,
        )

        self.pf_net.add_node(
            self.name + "_Inp_E_St",
            model="lif_danner",
            x=4.0 + anchor_x,
            y=0.0 + anchor_y,
            color="m",
            v0=-60.0,
            g_leak=5.0,
        )

    def add_connections(self):
        self.pf_net.add_edge(self.name + "_PF_F", self.name + "_Inp_F", weight=0.8)
        self.pf_net.add_edge(self.name + "_Inp_F", self.name + "_PF_E", weight=-1.5)
        self.pf_net.add_edge(self.name + "_Inp_F", self.name + "_PF_St", weight=-0.5)
        self.pf_net.add_edge(self.name + "_Inp_F", self.name + "_PF_Sw", weight=-0.1)

        self.pf_net.add_edge(self.name + "_PF_E", self.name + "_Inp_E", weight=1.0)
        self.pf_net.add_edge(self.name + "_Inp_E", self.name + "_PF_F", weight=-1.0)
        self.pf_net.add_edge(self.name + "_Inp_E", self.name + "_PF_Sw", weight=-0.25)
        self.pf_net.add_edge(self.name + "_Inp_E", self.name + "_PF_St", weight=-0.5)

        self.pf_net.add_edge(self.name + "_PF_Sw", self.name + "_Inp_Sw", weight=1.5)
        self.pf_net.add_edge(self.name + "_Inp_Sw", self.name + "_PF_St", weight=-2.0)
        self.pf_net.add_edge(self.name + "_Inp_Sw", self.name + "_PF_F", weight=-0.5)
        self.pf_net.add_edge(self.name + "_Inp_Sw", self.name + "_PF_E", weight=-0.5)

        self.pf_net.add_edge(self.name + "_PF_St", self.name + "_Inp_St", weight=1.5)
        self.pf_net.add_edge(self.name + "_Inp_St", self.name + "_PF_Sw", weight=-0.25)
        self.pf_net.add_edge(self.name + "_Inp_St", self.name + "_PF_E", weight=-2.0)
        self.pf_net.add_edge(self.name + "_Inp_St", self.name + "_PF_F", weight=-2.0)

        self.pf_net.add_edge(self.name + "_Inp_F_Sw", self.name + "_PF_Sw", weight=-3.0)
        self.pf_net.add_edge(self.name + "_Inp_E_St", self.name + "_PF_St", weight=-3.0)
        return


class ConnectPF2Commissural(object):
    """Connect a PF circuit with Commissural"""

    def __init__(self, pf_l, pf_r, comm_l, comm_r):
        """Initialization."""
        super(ConnectPF2Commissural, self).__init__()
        self.net = nx.compose_all([pf_l, pf_r, comm_l, comm_r])
        self.name = self.net.name[0] + "PF_COMM"

        # Methods
        self.connect_circuits()
        return

    def connect_circuits(self):
        """Connect PatternFormation to Commissural."""

        def _name(side, name):
            """Add the network name to the neuron."""
            return self.name[0] + side + "_" + name

        self.net.add_edge(_name("L", "V0D"), _name("L", "PF_F"), weight=-4.0)

        self.net.add_edge(_name("R", "V0D"), _name("R", "PF_F"), weight=-4.0)

        self.net.add_edge(_name("L", "InV0V"), _name("L", "PF_F"), weight=-3.0)

        self.net.add_edge(_name("R", "InV0V"), _name("R", "PF_F"), weight=-3.0)

        self.net.add_edge(_name("L", "V3"), _name("L", "PF_F"), weight=2.0)

        self.net.add_edge(_name("R", "V3"), _name("R", "PF_F"), weight=2.0)


class ConnectPF2RG(object):
    """Connect a PF circuit with RG"""

    def __init__(self, rg, pf):
        """Initialization."""
        super(ConnectPF2RG, self).__init__()
        self.net = nx.compose_all([rg, pf])
        self.name = self.net.name

        # Methods
        self.connect_circuits()
        return

    def connect_circuits(self):
        """Connect CPG's to Interneurons."""

        def _name(name):
            """Add the network name to the neuron."""
            return self.name + "_" + name

        self.net.add_edge(_name("RG_F"), _name("PF_F"), weight=1.0)
        self.net.add_edge(_name("RG_F"), _name("PF_Sw"), weight=0.6)
        self.net.add_edge(_name("RG_F"), _name("Inp_F_Sw"), weight=0.4)

        self.net.add_edge(_name("RG_E"), _name("PF_E"), weight=0.7)
        self.net.add_edge(_name("RG_E"), _name("PF_St"), weight=0.5)
        self.net.add_edge(_name("RG_E"), _name("Inp_E_St"), weight=0.375)

        self.net.add_edge(_name("In_F"), _name("PF_E"), weight=-1.5)
        self.net.add_edge(_name("In_E"), _name("PF_F"), weight=-1.5)

        return self.net


class ConnectMN2CPG(object):
    """Connect a PF circuit with RG"""

    def __init__(self, cpg, mn, mn_names_flex):
        """Initialization."""
        super(ConnectMN2CPG, self).__init__()
        self.net = nx.compose_all([cpg, mn])
        self.name = self.net.name
        self.mn_names = list(mn.nodes)
        self.mn_names_flex = mn_names_flex

        # Methods
        self.connect_circuits()
        return

    def connect_circuits(self):
        """Connect CPG's to Interneurons."""

        def _name(name):
            """Add the network name to the neuron."""
            return self.name + "_" + name

        for nm in self.mn_names:
            if "_Mn_" in nm:
                if any([mnf in nm for mnf in self.mn_names_flex]):
                    self.net.add_edge(_name("PF_F"), nm, weight=2.0)
                    self.net.add_edge(_name("PF_Sw"), nm, weight=2.0)
                    self.net.add_edge(_name("Inp_E"), nm, weight=-1.0)
                    self.net.add_edge(_name("Inp_St"), nm, weight=-1.0)
                else:
                    self.net.add_edge(_name("PF_E"), nm, weight=2.0)
                    self.net.add_edge(_name("PF_St"), nm, weight=2.0)
                    self.net.add_edge(_name("PF_F"), nm, weight=0.5)
                    self.net.add_edge(_name("PF_Sw"), nm, weight=0.5)
                    self.net.add_edge(_name("Inp_F"), nm, weight=-1.0)
                    self.net.add_edge(_name("Inp_Sw"), nm, weight=-1.0)
        return self.net


def main():
    """Main."""

    net = CPG("FORE")  # Directed graph
    nx.write_graphml(net.cpg, "./conf/auto_gen_danner_cpg_net.graphml")

    return


if __name__ == "__main__":
    main()
