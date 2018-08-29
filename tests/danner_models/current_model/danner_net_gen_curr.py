""" Generate Danner Network. Current Model"""

import networkx as nx
import numpy as np


class CPG(object):
    """Generate CPG Network
    """

    def __init__(self, name, anchor_x=0.0, anchor_y=0.0):
        """ Initialization. """
        super(CPG, self).__init__()
        self.cpg = nx.DiGraph()
        self.name = name
        self.cpg.name = name

        #: Methods
        self.add_neurons(anchor_x, anchor_y)
        self.add_connections()
        return

    def add_neurons(self, anchor_x, anchor_y):
        """ Add neurons. """
        self.cpg.add_node(self.name+'_RG_F',
                          model='lif_danner_nap',
                          x=1.0+anchor_x,
                          y=0.0+anchor_y,
                          color='r',
                          m_e=0.1,
                          b_e=0.0,
                          v0=-60.0,
                          h0=np.random.uniform(0, 1))
        self.cpg.add_node(self.name+'_RG_E',
                          model='lif_danner_nap',
                          x=1.0+anchor_x,
                          y=4.0+anchor_y,
                          color='b',
                          m_e=0.0,
                          b_e=0.1,
                          v0=-60.0,
                          h0=np.random.uniform(0, 1))
        self.cpg.add_node(self.name+'_In_F',
                          model='lif_danner',
                          x=0.0+anchor_x,
                          y=2.0+anchor_y,
                          color='m',
                          v0=-60.0)
        self.cpg.add_node(self.name+'_In_E',
                          model='lif_danner',
                          x=2.0+anchor_x,
                          y=2.0+anchor_y,
                          color='m',
                          v0=-60.0)

    def add_connections(self):
        self.cpg.add_edge(self.name+'_RG_F',
                          self.name+'_In_F',
                          weight=0.4)
        self.cpg.add_edge(self.name+'_In_F',
                          self.name+'_RG_E',
                          weight=-1.0)
        self.cpg.add_edge(self.name+'_RG_E',
                          self.name+'_In_E',
                          weight=0.4)
        self.cpg.add_edge(self.name+'_In_E',
                          self.name+'_RG_F',
                          weight=-0.08)
        return


class PatterFormation(object):
    """Pattern Formation Layer

    """

    def __init__(self, name, anchor_x=0.0, anchor_y=0.0, color='g'):
        super(PatterFormation, self).__init__()
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
                             x=1.0+anchor_x,
                             y=0.0+anchor_y,
                             color='r',
                             m_e=0.1,
                             b_e=0.0,
                             v0=-60.0,
                             h0=np.random.uniform(0, 1))

        self.pf_net.add_node(self.name+'_PF_E',
                             model='lif_danner_nap',
                             x=1.0+anchor_x,
                             y=4.0+anchor_y,
                             color='b',
                             m_e=0.0,
                             b_e=0.1,
                             v0=-60.0,
                             h0=np.random.uniform(0, 1))

        self.pf_net.add_node(self.name+'_Inp_F',
                             model='lif_danner',
                             x=0.0+anchor_x,
                             y=2.0+anchor_y,
                             color='m',
                             v0=-60.0)

        self.pf_net.add_node(self.name+'_Inp_E',
                             model='lif_danner',
                             x=2.0+anchor_x,
                             y=2.0+anchor_y,
                             color='m',
                             v0=-60.0)

        self.pf_net.add_node(self.name+'_PF_Sw',
                             model='lif_danner_nap',
                             x=1.0+anchor_x,
                             y=-4.0+anchor_y,
                             color='r',
                             m_e=0.1,
                             b_e=0.0,
                             v0=-60.0,
                             h0=np.random.uniform(0, 1))

        self.pf_net.add_node(self.name+'_PF_St',
                             model='lif_danner_nap',
                             x=1.0+anchor_x,
                             y=-8.0+anchor_y,
                             color='b',
                             m_e=0.0,
                             b_e=0.1,
                             v0=-60.0,
                             h0=np.random.uniform(0, 1))

        self.pf_net.add_node(self.name+'_Inp_Sw',
                             model='lif_danner',
                             x=0.0+anchor_x,
                             y=-6.0+anchor_y,
                             color='m',
                             v0=-60.0)

        self.pf_net.add_node(self.name+'_Inp_St',
                             model='lif_danner',
                             x=2.0+anchor_x,
                             y=-6.0+anchor_y,
                             color='m',
                             v0=-60.0)

    def add_connections(self):
        self.pf_net.add_edge(self.name+'_PF_F',
                             self.name+'_Inp_F',
                             weight=0.4)
        self.pf_net.add_edge(self.name+'_Inp_F',
                             self.name+'_PF_E',
                             weight=-1.0)
        self.pf_net.add_edge(self.name+'_PF_E',
                             self.name+'_Inp_E',
                             weight=0.4)
        self.pf_net.add_edge(self.name+'_Inp_E',
                             self.name+'_PF_F',
                             weight=-0.08)

        self.pf_net.add_edge(self.name+'_PF_Sw',
                             self.name+'_Inp_Sw',
                             weight=0.4)
        self.pf_net.add_edge(self.name+'_Inp_Sw',
                             self.name+'_PF_St',
                             weight=-1.0)
        self.pf_net.add_edge(self.name+'_PF_St',
                             self.name+'_Inp_St',
                             weight=0.4)
        self.pf_net.add_edge(self.name+'_Inp_St',
                             self.name+'_PF_Sw',
                             weight=-0.08)
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
        self.mn_net.add_node(self.name+'_Inp_St',
                             model='lif_danner',
                             x=2.0+anchor_x,
                             y=-6.0+anchor_y,
                             color='m',
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
        self.commissural = nx.DiGraph()
        self.name = name
        self.commissural.name = name

        #: Methods
        self.add_neurons(anchor_x, anchor_y, color)
        self.add_connections()
        return

    def add_neurons(self, anchor_x, anchor_y, color):
        """ Add neurons. """
        self.commissural.add_node(self.name+'_CINi1',
                                  model='lif_danner',
                                  x=1.0+anchor_x,
                                  y=-1.0+anchor_y,
                                  color='m',
                                  v0=-60.0)
        self.commissural.add_node(self.name+'_CINe1',
                                  model='lif_danner',
                                  x=1.0+anchor_x,
                                  y=1.0+anchor_y,
                                  color='g',
                                  m_e=0.15,
                                  b_e=0.0,
                                  v0=-60.0)
        self.commissural.add_node(self.name+'_CINi2',
                                  model='lif_danner',
                                  x=1.0+anchor_x,
                                  y=3.0+anchor_y,
                                  color='m',
                                  m_e=0.75,
                                  b_e=0.0,
                                  v0=-60.0)
        self.commissural.add_node(self.name+'_CINe2',
                                  model='lif_danner',
                                  x=1.0+anchor_x,
                                  y=5.0+anchor_y,
                                  color='g',
                                  v0=-60.0)
        self.commissural.add_node(self.name+'_CINe3',
                                  model='lif_danner',
                                  x=(-1.0 if self.name[-1] ==
                                     'L' else 3.0)+anchor_x,
                                  y=1.0+anchor_y,
                                  color='g',
                                  v0=-60.0)
        self.commissural.add_node(self.name+'_CINe4',
                                  x=(-1.0 if self.name[-1] ==
                                     'L' else 3.0)+anchor_x,
                                  model='lif_danner',
                                  y=6.0+anchor_y,
                                  color='g',
                                  v0=-60.0)
        self.commissural.add_node(self.name+'_Ini1',
                                  model='lif_danner',
                                  x=(-1.0 if self.name[-1] ==
                                     'L' else 3.0)+anchor_x,
                                  y=3+anchor_y,
                                  color=color,
                                  v0=-60.0)
        return

    def add_connections(self):
        return


class LPSN(object):
    """Long Propriospinal Neuron Network template.

    """

    def __init__(self, name, anchor_x=0.0, anchor_y=0.0, color='c'):
        """ Initialization. """
        super(LPSN, self).__init__()
        self.lpsn = nx.DiGraph()
        self.name = name
        self.lpsn.name = name

        #: Methods
        self.add_neurons(anchor_x, anchor_y, color)
        self.add_connections()
        return

    def add_neurons(self, anchor_x, anchor_y, color):
        """ Add neurons. """
        self.lpsn.add_node(self.name+'_LPNi1',
                           model='lif_danner',
                           x=1.0+anchor_x,
                           y=0.0+anchor_y,
                           color='m',
                           m_e=0.75,
                           b_e=0.0,
                           v0=-60.0)
        self.lpsn.add_node(self.name+'_LPNe1',
                           model='lif_danner',
                           x=1.0+anchor_x,
                           y=2+anchor_y,
                           color='g',
                           m_e=0.15,
                           b_e=0.0,
                           v0=-60.0)
        self.lpsn.add_node(self.name+'_LPNe2',
                           model='lif_danner',
                           x=1.0+anchor_x,
                           y=4+anchor_y,
                           color='g',
                           v0=-60.0)
        self.lpsn.add_node(self.name+'_LPNi',
                           model='lif_danner',
                           x=(-2.0 if self.name[-1] ==
                              'L' else 4.0)+anchor_x,
                           y=4+anchor_y,
                           color='g',
                           v0=-60.0)
        self.lpsn.add_node(self.name+'_LPNsh1',
                           model='lif_danner',
                           x=(-4.0 if self.name[-1] ==
                              'L' else 6.0)+anchor_x,
                           y=0+anchor_y,
                           color='g',
                           v0=-60.0)
        self.lpsn.add_node(self.name+'_LPNsh2',
                           model='lif_danner',
                           x=(-4.0 if self.name[-1] ==
                              'L' else 6.0)+anchor_x,
                           y=4+anchor_y,
                           color='g',
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

        self.net.add_edge(_name('L', 'RG_E'),
                          _name('L', 'CINi1'),
                          weight=0.40)

        self.net.add_edge(_name('R', 'RG_E'),
                          _name('R', 'CINi1'),
                          weight=0.40)

        self.net.add_edge(_name('L', 'RG_F'),
                          _name('L', 'CINe3'),
                          weight=1.0)

        self.net.add_edge(_name('R', 'RG_F'),
                          _name('R', 'CINe3'),
                          weight=1.0)

        self.net.add_edge(_name('L', 'RG_F'),
                          _name('L', 'CINi2'),
                          weight=0.70)

        self.net.add_edge(_name('R', 'RG_F'),
                          _name('R', 'CINi2'),
                          weight=0.70)

        self.net.add_edge(_name('L', 'RG_F'),
                          _name('L', 'CINe2'),
                          weight=0.35)

        self.net.add_edge(_name('R', 'RG_F'),
                          _name('R', 'CINe2'),
                          weight=0.35)

        self.net.add_edge(_name('L', 'RG_F'),
                          _name('L', 'CINe4'),
                          weight=0.50)

        self.net.add_edge(_name('R', 'RG_F'),
                          _name('R', 'CINe4'),
                          weight=0.50)

        self.net.add_edge(_name('L', 'CINi1'),
                          _name('R', 'RG_F'),
                          weight=-0.03)

        self.net.add_edge(_name('R', 'CINi1'),
                          _name('L', 'RG_F'),
                          weight=-0.03)

        self.net.add_edge(_name('L', 'CINe1'),
                          _name('R', 'Ini1'),
                          weight=0.60)

        self.net.add_edge(_name('R', 'CINe1'),
                          _name('L', 'Ini1'),
                          weight=0.60)

        self.net.add_edge(_name('L', 'CINi2'),
                          _name('R', 'RG_F'),
                          weight=-0.07)

        self.net.add_edge(_name('R', 'CINi2'),
                          _name('L', 'RG_F'),
                          weight=-0.07)

        self.net.add_edge(_name('L', 'CINe2'),
                          _name('R', 'RG_F'),
                          weight=0.03)

        self.net.add_edge(_name('R', 'CINe2'),
                          _name('L', 'RG_F'),
                          weight=0.03)

        self.net.add_edge(_name('L', 'CINe3'),
                          _name('L', 'CINe1'),
                          weight=1.0)

        self.net.add_edge(_name('R', 'CINe3'),
                          _name('R', 'CINe1'),
                          weight=1.0)

        self.net.add_edge(_name('L', 'Ini1'),
                          _name('L', 'RG_F'),
                          weight=-0.07)

        self.net.add_edge(_name('R', 'Ini1'),
                          _name('R', 'RG_F'),
                          weight=-0.07)

        return self.net


class ConnectFore2Hind(object):
    """Connect a Fore limb circuit with Hind Limb
    """

    def __init__(self, fore, hind, lspn_l, lspn_r):
        """ Initialization. """
        super(ConnectFore2Hind, self).__init__()
        self.net = nx.compose_all([fore,
                                   hind,
                                   lspn_l,
                                   lspn_r])
        self.name = self.net.name[0] + 'MODEL'

        #: Methods
        self.connect_circuits()
        return

    def connect_circuits(self):
        """ Connect CPG's to Interneurons. """

        def _name(side, name, f_h=''):
            """ Add the network name to the neuron."""
            return f_h + side + '_' + name

        self.net.add_edge(_name('L', 'RG_E', 'F'),
                          _name('L', 'LPNsh1'),
                          weight=0.50)

        self.net.add_edge(_name('R', 'RG_E', 'F'),
                          _name('R', 'LPNsh1'),
                          weight=0.50)

        self.net.add_edge(_name('L', 'RG_E', 'H'),
                          _name('L', 'LPNsh2'),
                          weight=0.50)

        self.net.add_edge(_name('R', 'RG_E', 'H'),
                          _name('R', 'LPNsh2'),
                          weight=0.50)

        self.net.add_edge(_name('L', 'RG_F', 'F'),
                          _name('L', 'LPNi'),
                          weight=0.70)

        self.net.add_edge(_name('R', 'RG_F', 'F'),
                          _name('R', 'LPNi'),
                          weight=0.70)

        self.net.add_edge(_name('L', 'RG_F', 'F'),
                          _name('L', 'LPNi1'),
                          weight=0.50)

        self.net.add_edge(_name('R', 'RG_F', 'F'),
                          _name('R', 'LPNi1'),
                          weight=0.50)

        self.net.add_edge(_name('L', 'CINe4', 'F'),
                          _name('L', 'LPNe1'),
                          weight=0.90)

        self.net.add_edge(_name('R', 'CINe4', 'F'),
                          _name('R', 'LPNe1'),
                          weight=0.90)

        self.net.add_edge(_name('L', 'CINe4', 'H'),
                          _name('L', 'LPNe2'),
                          weight=0.90)

        self.net.add_edge(_name('R', 'CINe4', 'H'),
                          _name('R', 'LPNe2'),
                          weight=0.90)

        self.net.add_edge(_name('L', 'LPNi1'),
                          _name('R', 'RG_F', 'H'),
                          weight=-0.01)

        self.net.add_edge(_name('R', 'LPNi1'),
                          _name('L', 'RG_F', 'H'),
                          weight=-0.01)

        self.net.add_edge(_name('L', 'LPNe1'),
                          _name('R', 'RG_F', 'H'),
                          weight=0.60)

        self.net.add_edge(_name('R', 'LPNe1'),
                          _name('L', 'RG_F', 'H'),
                          weight=0.60)

        self.net.add_edge(_name('L', 'LPNe2'),
                          _name('R', 'RG_F', 'F'),
                          weight=0.60)

        self.net.add_edge(_name('R', 'LPNe2'),
                          _name('L', 'RG_F', 'F'),
                          weight=0.60)

        self.net.add_edge(_name('L', 'LPNi'),
                          _name('L', 'RG_F', 'H'),
                          weight=-0.01)

        self.net.add_edge(_name('R', 'LPNi'),
                          _name('R', 'RG_F', 'H'),
                          weight=-0.01)

        self.net.add_edge(_name('L', 'LPNsh1'),
                          _name('L', 'RG_F', 'H'),
                          weight=0.01)

        self.net.add_edge(_name('R', 'LPNsh1'),
                          _name('R', 'RG_F', 'H'),
                          weight=0.01)

        self.net.add_edge(_name('L', 'LPNsh2'),
                          _name('L', 'RG_F', 'F'),
                          weight=0.125)

        self.net.add_edge(_name('R', 'LPNsh2'),
                          _name('R', 'RG_F', 'F'),
                          weight=0.125)

        return self.net


def main():
    """ Main. """

    net = CPG('FORE')  #: Directed graph
    nx.write_graphml(net.cpg, './conf/auto_gen_danner_cpg.graphml')

    return


if __name__ == '__main__':
    main()
