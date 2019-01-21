#!/usr/bin/env python

""" Generate Danner Network. Current Model"""

import networkx as nx
import numpy as np


class CPG(object):
    """Generate CPG Network
    """

    def __init__(self, name, anchor_x=0.0, anchor_y=0.0):
        """ Initialization. """
        super(CPG, self).__init__()
        self.cpg_net = nx.DiGraph()
        self.name = name
        self.cpg_net.name = name

        #: Methods
        self.add_neurons(anchor_x, anchor_y)
        self.add_connections()
        return

    def add_neurons(self, anchor_x, anchor_y):
        """ Add neurons. """
        self.cpg_net.add_node(self.name+'_RG_F',
                              model='lif_danner_nap',
                              x=1.0+anchor_x,
                              y=0.0+anchor_y,
                              color='r',
                              m_e=0.1,
                              b_e=0.0,
                              v0=-60.0,
                              h0=np.random.uniform(0, 1))
        self.cpg_net.add_node(self.name+'_RG_E',
                              model='lif_danner_nap',
                              x=1.0+anchor_x,
                              y=4.0+anchor_y,
                              color='b',
                              m_e=0.0,
                              b_e=0.1,
                              v0=-60.0,
                              h0=np.random.uniform(0, 1))
        self.cpg_net.add_node(self.name+'_In_F',
                              model='lif_danner',
                              x=0.0+anchor_x,
                              y=2.0+anchor_y,
                              color='m',
                              v0=-60.0)
        self.cpg_net.add_node(self.name+'_In_E',
                              model='lif_danner',
                              x=2.0+anchor_x,
                              y=2.0+anchor_y,
                              color='m',
                              v0=-60.0)

    def add_connections(self):
        self.cpg_net.add_edge(self.name+'_RG_F',
                              self.name+'_In_F',
                              weight=0.4)
        self.cpg_net.add_edge(self.name+'_In_F',
                              self.name+'_RG_E',
                              weight=-1.0)
        self.cpg_net.add_edge(self.name+'_RG_E',
                              self.name+'_In_E',
                              weight=0.4)
        self.cpg_net.add_edge(self.name+'_In_E',
                              self.name+'_RG_F',
                              weight=-0.08)
        return


class PatternFormation(object):
    """Pattern Formation Layer

    """

    def __init__(self, name, anchor_x=0.0, anchor_y=0.0, color='g'):
        super(PatternFormation, self).__init__()
        self.name = name
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y
        self.color = color

        self.pf_net = nx.DiGraph()

        #: Methods
        self.add_neurons(anchor_x, anchor_y, color)
        self.add_connections()

        return

    def add_neurons(self, anchor_x, anchor_y, color):
        """ Add neurons. """
        self.pf_net.add_node(self.name+'_PF_F',
                             model='lif_danner_nap',
                             x=-3.0+anchor_x,
                             y=0.0+anchor_y,
                             color='r',
                             m_e=0.1,
                             b_e=0.0,
                             v0=-60.0,
                             h0=np.random.uniform(0, 1))

        self.pf_net.add_node(self.name+'_PF_E',
                             model='lif_danner_nap',
                             x=-3.0+anchor_x,
                             y=4.0+anchor_y,
                             color='b',
                             m_e=0.0,
                             b_e=0.1,
                             v0=-60.0,
                             h0=np.random.uniform(0, 1))

        self.pf_net.add_node(self.name+'_Inp_F',
                             model='lif_danner',
                             x=-4.0+anchor_x,
                             y=2.0+anchor_y,
                             color='m',
                             v0=-60.0)

        self.pf_net.add_node(self.name+'_Inp_E',
                             model='lif_danner',
                             x=-2.0+anchor_x,
                             y=2.0+anchor_y,
                             color='m',
                             v0=-60.0)

        self.pf_net.add_node(self.name+'_PF_Sw',
                             model='lif_danner_nap',
                             x=5.0+anchor_x,
                             y=0.0+anchor_y,
                             color='r',
                             m_e=0.1,
                             b_e=0.0,
                             v0=-60.0,
                             h0=np.random.uniform(0, 1))

        self.pf_net.add_node(self.name+'_PF_St',
                             model='lif_danner_nap',
                             x=5.0+anchor_x,
                             y=4.0+anchor_y,
                             color='b',
                             m_e=0.0,
                             b_e=0.1,
                             v0=-60.0,
                             h0=np.random.uniform(0, 1))

        self.pf_net.add_node(self.name+'_Inp_Sw',
                             model='lif_danner',
                             x=6.0+anchor_x,
                             y=2.0+anchor_y,
                             color='m',
                             v0=-60.0)

        self.pf_net.add_node(self.name+'_Inp_St',
                             model='lif_danner',
                             x=4.0+anchor_x,
                             y=2.0+anchor_y,
                             color='m',
                             v0=-60.0)

        self.pf_net.add_node(self.name+'_Inp_F_Sw',
                             model='lif_danner',
                             x=6.0+anchor_x,
                             y=0.0+anchor_y,
                             color='m',
                             v0=-60.0)

        self.pf_net.add_node(self.name+'_Inp_E_St',
                             model='lif_danner',
                             x=4.0+anchor_x,
                             y=0.0+anchor_y,
                             color='m',
                             v0=-60.0)

    def add_connections(self):
        self.pf_net.add_edge(self.name+'_PF_F',
                             self.name+'_Inp_F',
                             weight=8.)
        self.pf_net.add_edge(self.name+'_Inp_F',
                             self.name+'_PF_E',
                             weight=-15.)
        self.pf_net.add_edge(self.name+'_PF_E',
                             self.name+'_Inp_E',
                             weight=10.)
        self.pf_net.add_edge(self.name+'_Inp_E',
                             self.name+'_PF_F',
                             weight=-10.)

        self.pf_net.add_edge(self.name+'_PF_Sw',
                             self.name+'_Inp_Sw',
                             weight=15.)
        self.pf_net.add_edge(self.name+'_Inp_Sw',
                             self.name+'_PF_St',
                             weight=-20.)
        self.pf_net.add_edge(self.name+'_PF_St',
                             self.name+'_Inp_St',
                             weight=15.)
        self.pf_net.add_edge(self.name+'_Inp_St',
                             self.name+'_PF_Sw',
                             weight=-2.5)
        return


class Motorneurons(object):
    """Motorneurons layers. Also contains interneurons.

    """

    def __init__(self, name, anchor_x=0.0, anchor_y=0.0, color='r'):
        super(Motorneurons, self).__init__()
        self.name = name
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y
        self.color = color

        self.mn_net = nx.DiGraph()

        return

        #: Methods
        self.add_neurons(anchor_x, anchor_y, color)
        self.add_connections()

        return

    def add_neurons(self, anchor_x, anchor_y, color):
        """ Add neurons. """
        biolog.debug("Adding motorneurons")
        self.mn_net.add_node(self.name+'_InIaSw',
                             model='lif_danner',
                             x=2.0+anchor_x,
                             y=6.0+anchor_y,
                             color='k',
                             v0=-60.0)

        self.mn_net.add_node(self.name+'_InIaSt',
                             model='lif_danner',
                             x=0.0+anchor_x,
                             y=6.0+anchor_y,
                             color='k',
                             v0=-60.0)

        self.mn_net.add_node(self.name+'_InIaF',
                             model='lif_danner',
                             x=-2.0+anchor_x,
                             y=6.0+anchor_y,
                             color='k',
                             v0=-60.0)

        self.mn_net.add_node(self.name+'_InIaE',
                             model='lif_danner',
                             x=-4.0+anchor_x,
                             y=6.0+anchor_y,
                             color='k',
                             v0=-60.0)

    def add_connections(self):
        """ Connect the neurons."""
        pass


class Commissural(object):
    """Commissural Network template.

    """

    def __init__(self, name, anchor_x=0.0, anchor_y=0.0, color='y'):
        """ Initialization. """
        super(Commissural, self).__init__()
        self.comm_net = nx.DiGraph()
        self.name = name
        self.comm_net.name = name

        #: Methods
        self.add_neurons(anchor_x, anchor_y, color)
        self.add_connections()
        return

    def add_neurons(self, anchor_x, anchor_y, color):
        """ Add neurons. """
        self.comm_net.add_node(self.name+'_V0D',
                               model='lif_danner',
                               x=1.0+anchor_x,
                               y=-1.0+anchor_y,
                               color='m',
                               v0=-60.0)
        self.comm_net.add_node(self.name+'_V3',
                               model='lif_danner',
                               x=1.0+anchor_x,
                               y=1.0+anchor_y,
                               color='g',
                               m_e=0.15,
                               b_e=0.0,
                               v0=-60.0)
        self.comm_net.add_node(self.name+'_V0V',
                               model='lif_danner',
                               x=1.0+anchor_x,
                               y=3.0+anchor_y,
                               color='m',
                               m_e=0.75,
                               b_e=0.0,
                               v0=-60.0)
        self.comm_net.add_node(self.name+'_InV0V',
                               model='lif_danner',
                               x=1.0+anchor_x,
                               y=5.0+anchor_y,
                               color='g',
                               v0=-60.0)
        self.comm_net.add_node(self.name+'_V0D2',
                               model='lif_danner',
                               x=(-1.0 if self.name[-1] ==
                                     'L' else 3.0)+anchor_x,
                               y=1.0+anchor_y,
                               color='g',
                               v0=-60.0)
        self.comm_net.add_node(self.name+'_V3e',
                               x=(-1.0 if self.name[-1] ==
                                  'L' else 3.0)+anchor_x,
                               model='lif_danner',
                               y=6.0+anchor_y,
                               color='g',
                               v0=-60.0)
        return

    def add_connections(self):
        return


class HomoLateral(object):
    """Homo-lateral connections
    """

    def __init__(self, name, anchor_x=0.0, anchor_y=0.0, color='c'):
        """ Initialization. """
        super(HomoLateral, self).__init__()
        self.homo_net = nx.DiGraph()
        self.name = name
        self.homo_net.name = name

        #: Methods
        self.add_neurons(anchor_x, anchor_y, color)
        self.add_connections()
        return

    def add_neurons(self, anchor_x, anchor_y, color):
        """ Add neurons. """
        self.homo_net.add_node('H'+self.name+'_V2aHom',
                               model='lif_danner',
                               x=1.0+anchor_x,
                               y=0.0+anchor_y,
                               color='m',
                               m_e=0.75,
                               b_e=0.0,
                               v0=-60.0)
        self.homo_net.add_node('F'+self.name+'_V2aHom',
                               model='lif_danner',
                               x=1.0+anchor_x,
                               y=0.0+anchor_y,
                               color='m',
                               m_e=0.75,
                               b_e=0.0,
                               v0=-60.0)
        self.homo_net.add_node(self.name+'_InFront',
                               model='lif_danner',
                               x=1.0+anchor_x,
                               y=2+anchor_y,
                               color='g',
                               m_e=0.15,
                               b_e=0.0,
                               v0=-60.0)
        return

    def add_connections(self):
        return


class ContraLateral(object):
    """Contra-lateral connections
    """

    def __init__(self, name, anchor_x=0.0, anchor_y=0.0, color='c'):
        """ Initialization. """
        super(ContraLateral, self).__init__()
        self.contra_net = nx.DiGraph()
        self.name = name
        self.contra_net.name = name

        #: Methods
        self.add_neurons(anchor_x, anchor_y, color)
        self.add_connections()
        return

    def add_neurons(self, anchor_x, anchor_y, color):
        """ Add neurons. """
        self.contra_net.add_node(self.name+'_V0V_diag',
                                 model='lif_danner',
                                 x=1.0+anchor_x,
                                 y=0.0+anchor_y,
                                 color='y',
                                 m_e=0.75,
                                 b_e=0.0,
                                 v0=-60.0)
        self.contra_net.add_node(self.name+'_V0D_diag',
                                 model='lif_danner',
                                 x=1.0+anchor_x,
                                 y=0.0+anchor_y,
                                 color='y',
                                 m_e=0.75,
                                 b_e=0.0,
                                 v0=-60.0)
        self.contra_net.add_node(self.name+'_V2aV0V',
                                 model='lif_danner',
                                 x=1.0+anchor_x,
                                 y=2.+anchor_y,
                                 color='y',
                                 m_e=0.15,
                                 b_e=0.0,
                                 v0=-60.0)
        self.contra_net.add_node(self.name+'_V2aV0V_diag',
                                 model='lif_danner',
                                 x=1.0+anchor_x,
                                 y=3.+anchor_y,
                                 color='y',
                                 m_e=0.15,
                                 b_e=0.0,
                                 v0=-60.0)
        self.contra_net.add_node(self.name+'_V3e',
                                 model='lif_danner',
                                 x=1.0+anchor_x,
                                 y=2.+anchor_y,
                                 color='y',
                                 m_e=0.15,
                                 b_e=0.0,
                                 v0=-60.0)
        return

    def add_connections(self):
        return


class ConnectRG2Commissural(object):
    """Connect a RG circuit with Commissural
    """

    def __init__(self, rg_l, rg_r, comm_l, comm_r):
        """ Initialization. """
        super(ConnectRG2Commissural, self).__init__()
        self.net = nx.compose_all([rg_l,
                                   rg_r,
                                   comm_l,
                                   comm_r])
        self.name = self.net.name[0] + 'RG_COMM'

        #: Methods
        self.connect_circuits()
        return

    def connect_circuits(self):
        """ Connect CPG's to Interneurons. """

        def _name(side, name):
            """ Add the network name to the neuron."""
            return self.name[0] + side + '_' + name

        self.net.add_edge(_name('L', 'RG_F'),
                          _name('L', 'V0D'),
                          weight=10.)

        self.net.add_edge(_name('R', 'RG_F'),
                          _name('R', 'V0D'),
                          weight=10.)

        self.net.add_edge(_name('L', 'RG_F'),
                          _name('L', 'V3'),
                          weight=10.)

        self.net.add_edge(_name('R', 'RG_F'),
                          _name('R', 'V3'),
                          weight=10.)

        self.net.add_edge(_name('L', 'V0D'),
                          _name('L', 'RG_F'),
                          weight=-10.)

        self.net.add_edge(_name('R', 'V0D'),
                          _name('R', 'RG_F'),
                          weight=-10.)

        self.net.add_edge(_name('L', 'V0D2'),
                          _name('L', 'RG_F'),
                          weight=-10.)

        self.net.add_edge(_name('R', 'V0D2'),
                          _name('R', 'RG_F'),
                          weight=-10.)

        self.net.add_edge(_name('L', 'V0V'),
                          _name('L', 'InV0V'),
                          weight=6.)

        self.net.add_edge(_name('R', 'V0V'),
                          _name('R', 'InV0V'),
                          weight=6.)

        self.net.add_edge(_name('L', 'V3'),
                          _name('L', 'RG_F'),
                          weight=10.)

        self.net.add_edge(_name('R', 'V3'),
                          _name('R', 'RG_F'),
                          weight=10.)

        self.net.add_edge(_name('L', 'RG_E'),
                          _name('L', 'V0D2'),
                          weight=10.)

        self.net.add_edge(_name('R', 'RG_E'),
                          _name('R', 'V0D2'),
                          weight=10.)

        self.net.add_edge(_name('L', 'RG_E'),
                          _name('L', 'V3e'),
                          weight=10.)

        self.net.add_edge(_name('R', 'RG_E'),
                          _name('R', 'V3e'),
                          weight=10.)
        return self.net


class ConnectPF2Commissural(object):
    """Connect a PF circuit with Commissural
    """

    def __init__(self, pf_l, pf_r, comm_l, comm_r):
        """ Initialization. """
        super(ConnectPF2Commissural, self).__init__()
        self.net = nx.compose_all([pf_l,
                                   pf_r,
                                   comm_l,
                                   comm_r])
        self.name = self.net.name[0] + 'PF_COMM'

        #: Methods
        self.connect_circuits()
        return

    def connect_circuits(self):
        """ Connect PatternFormation to Commissural. """

        def _name(side, name):
            """ Add the network name to the neuron."""
            return self.name[0] + side + '_' + name

        self.net.add_edge(_name('L', 'V0D'),
                          _name('L', 'PF_F'),
                          weight=-4.)

        self.net.add_edge(_name('R', 'V0D'),
                          _name('R', 'PF_F'),
                          weight=-4.)

        self.net.add_edge(_name('L', 'InV0V'),
                          _name('L', 'PF_F'),
                          weight=-3.)

        self.net.add_edge(_name('R', 'InV0V'),
                          _name('R', 'PF_F'),
                          weight=-3.)

        self.net.add_edge(_name('L', 'V3'),
                          _name('L', 'PF_F'),
                          weight=2.)

        self.net.add_edge(_name('R', 'V3'),
                          _name('R', 'PF_F'),
                          weight=2.)


class ConnectFore2Hind(object):
    """Connect a Fore limb circuit with Hind Limb
    """

    def __init__(self, fore, hind, homo, contra):
        """ Initialization. """
        super(ConnectFore2Hind, self).__init__()
        self.net = nx.compose_all([fore,
                                   hind,
                                   homo,
                                   contra])
        self.name = self.net.name[0] + 'MODEL'

        #: Methods
        self.connect_circuits()
        return

    def connect_circuits(self):
        """ Connect CPG's to Interneurons. """

        def _name(side, name, f_h=''):
            """ Add the network name to the neuron."""
            return f_h + side + '_' + name

        self.net.add_edge(_name('L', 'RG_F', 'F'),
                          _name('L', 'InFront'),
                          weight=7.)

        self.net.add_edge(_name('R', 'RG_F', 'F'),
                          _name('R', 'InFront'),
                          weight=7.)

        self.net.add_edge(_name('L', 'InFront'),
                          _name('L', 'RG_F', 'H'),
                          weight=-7.)

        self.net.add_edge(_name('R', 'InFront'),
                          _name('R', 'RG_F', 'H'),
                          weight=-7.)

        self.net.add_edge(_name('L', 'RG_E', 'H'),
                          _name('L', 'V2aHom', 'F'),
                          weight=5.)

        self.net.add_edge(_name('R', 'RG_E', 'H'),
                          _name('R', 'V2aHom', 'F'),
                          weight=5.)

        self.net.add_edge(_name('L', 'V2aHom', 'H'),
                          _name('L', 'RG_F', 'F'),
                          weight=5.)

        self.net.add_edge(_name('R', 'V2aHom', 'H'),
                          _name('R', 'RG_F', 'F'),
                          weight=5.)

        self.net.add_edge(_name('L', 'RG_F', 'F'),
                          _name('L', 'V2aV0V_diag', 'F'),
                          weight=5.)

        self.net.add_edge(_name('R', 'RG_F', 'F'),
                          _name('R', 'V2aV0V_diag', 'F'),
                          weight=5.)

        self.net.add_edge(_name('L', 'RG_F', 'H'),
                          _name('L', 'V2aV0V_diag', 'H'),
                          weight=5.)

        self.net.add_edge(_name('R', 'RG_F', 'H'),
                          _name('R', 'V2aV0V_diag', 'H'),
                          weight=5.)

        self.net.add_edge(_name('L', 'V2aV0V_diag', 'F'),
                          _name('L', 'V0V_diag', 'F'),
                          weight=5.)

        self.net.add_edge(_name('R', 'V2aV0V_diag', 'F'),
                          _name('R', 'V0V_diag', 'F'),
                          weight=5.)

        self.net.add_edge(_name('L', 'V2aV0V_diag', 'H'),
                          _name('L', 'V0V_diag', 'H'),
                          weight=5.)

        self.net.add_edge(_name('R', 'V2aV0V_diag', 'H'),
                          _name('R', 'V0V_diag', 'H'),
                          weight=5.)

        self.net.add_edge(_name('L', 'V0V_diag', 'F'),
                          _name('R', 'RG_F', 'H'),
                          weight=5.)

        self.net.add_edge(_name('R', 'V0V_diag', 'F'),
                          _name('L', 'RG_F', 'H'),
                          weight=5.)

        self.net.add_edge(_name('L', 'V0V_diag', 'H'),
                          _name('R', 'RG_F', 'F'),
                          weight=5.)

        self.net.add_edge(_name('R', 'V0V_diag', 'H'),
                          _name('L', 'RG_F', 'F'),
                          weight=5.)

        self.net.add_edge(_name('L', 'RG_F', 'F'),
                          _name('L', 'V0D_diag', 'F'),
                          weight=5.)

        self.net.add_edge(_name('R', 'RG_F', 'F'),
                          _name('R', 'V0D_diag', 'F'),
                          weight=5.)

        self.net.add_edge(_name('L', 'V0D_diag', 'F'),
                          _name('R', 'RG_F', 'H'),
                          weight=-5.)

        self.net.add_edge(_name('R', 'V0D_diag', 'F'),
                          _name('L', 'RG_F', 'H'),
                          weight=5.)

        return self.net


class ConnectPF2RG(object):
    """Connect a PF circuit with RG"""

    def __init__(self, rg, pf):
        """ Initialization. """
        super(ConnectPF2RG, self).__init__()
        self.net = nx.compose_all([rg,
                                   pf])
        self.name = self.net.name

        #: Methods
        self.connect_circuits()
        return

    def connect_circuits(self):
        """ Connect CPG's to Interneurons. """

        def _name(name):
            """ Add the network name to the neuron."""
            return self.name + '_' + name

        self.net.add_edge(_name('RG_F'),
                          _name('PF_F'),
                          weight=10.)

        self.net.add_edge(_name('RG_E'),
                          _name('PF_E'),
                          weight=7.)

        self.net.add_edge(_name('RG_F'),
                          _name('PF_Sw'),
                          weight=6.)

        self.net.add_edge(_name('RG_E'),
                          _name('PF_St'),
                          weight=5.)

        self.net.add_edge(_name('In_F'),
                          _name('PF_E'),
                          weight=-15.)

        self.net.add_edge(_name('In_E'),
                          _name('PF_F'),
                          weight=-15.)

        self.net.add_edge(_name('RG_F'),
                          _name('Inp_F_Sw'),
                          weight=4.)

        self.net.add_edge(_name('RG_E'),
                          _name('Inp_E_St'),
                          weight=3.75)

        self.net.add_edge(_name('Inp_F_Sw'),
                          _name('PF_Sw'),
                          weight=-30.)

        self.net.add_edge(_name('Inp_E_St'),
                          _name('PF_St'),
                          weight=-30.)

        return self.net


def main():
    """ Main. """

    net = CPG('FORE')  #: Directed graph
    nx.write_graphml(net.cpg_net, './conf/auto_gen_danner_cpg_net.graphml')

    return


if __name__ == '__main__':
    main()
