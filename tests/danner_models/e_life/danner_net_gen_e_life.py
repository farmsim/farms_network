""" Generate Danner Network. eLife"""

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
                          v0=-62.5,
                          h0=np.random.uniform(0, 1))
        self.cpg.add_node(self.name+'_RG_E',
                          model='lif_danner_nap',
                          x=1.0+anchor_x,
                          y=4.0+anchor_y,
                          color='b',
                          m_e=0.0,
                          b_e=0.1,
                          v0=-62.5,
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
        self.commissural.add_node(self.name+'_V2a_diag',
                            x=(-1.0 if self.name[-1] ==
                                'L' else 3.0)+anchor_x,
                            model='lif_danner',
                            y=6.0+anchor_y,
                            color='g',
                            v0=-60.0)
        self.commissural.add_node(self.name+'_CINi1',
                                  model='lif_danner',
                                  x=1.0+anchor_x,
                                  y=-1.0+anchor_y,
                                  color='m',
                                  v0=-60.0)
        self.commissural.add_node(self.name+'_V0V',
                                  model='lif_danner',
                                  x=1.0+anchor_x,
                                  y=1.0+anchor_y,
                                  color='g',
                                  m_i=0.15,
                                  b_i=0.0,
                                  v0=-60.0)
        self.commissural.add_node(self.name+'_V0D',
                                  model='lif_danner',
                                  x=1.0+anchor_x,
                                  y=3.0+anchor_y,
                                  color='m',
                                  m_i=0.75,
                                  b_i=0.0,
                                  v0=-60.0)
        self.commissural.add_node(self.name+'_V3',
                                  model='lif_danner',
                                  x=1.0+anchor_x,
                                  y=5.0+anchor_y,
                                  color='g',
                                  v0=-60.0)
        self.commissural.add_node(self.name+'_V2a',
                                  model='lif_danner',
                                  x=(-1.0 if self.name[-1] ==
                                     'L' else 3.0)+anchor_x,
                                  y=1.0+anchor_y,
                                  color='g',
                                  v0=-60.0)
        self.commissural.add_node(self.name+'_IniV0V',
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
        self.lpsn.add_node(self.name+'_V0D_diag',
                           model='lif_danner',
                           x=1.0+anchor_x,
                           y=0.0+anchor_y,
                           color='m',
                           m_i=0.75,
                           b_i=0.0,
                           v0=-60.0)
        self.lpsn.add_node(self.name+'_V0V_diag_fh',
                           model='lif_danner',
                           x=1.0+anchor_x,
                           y=2+anchor_y,
                           color='g',
                           v0=-60.0)
        self.lpsn.add_node(self.name+'_V0V_diag_hf',
                           model='lif_danner',
                           x=1.0+anchor_x,
                           y=4+anchor_y,
                           color='g',
                           v0=-60.0)
        self.lpsn.add_node(self.name+'_Ini_hom_fh',
                           model='lif_danner',
                           x=(-2.0 if self.name[-1] ==
                              'L' else 4.0)+anchor_x,
                           y=4+anchor_y,
                           color='g',
                           v0=-60.0)
        self.lpsn.add_node(self.name+'_Sh2_hom_fh',
                           model='lif_danner',
                           x=(-4.0 if self.name[-1] ==
                              'L' else 6.0)+anchor_x,
                           y=0+anchor_y,
                           color='g',
                           v0=-60.0)
        self.lpsn.add_node(self.name+'_Sh2_hom_hf',
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
                          _name('L', 'V2a'),
                          weight=1.0)

        self.net.add_edge(_name('R', 'RG_F'),
                          _name('R', 'V2a'),
                          weight=1.0)

        self.net.add_edge(_name('L', 'RG_F'),
                          _name('L', 'V0D'),
                          weight=0.70)

        self.net.add_edge(_name('R', 'RG_F'),
                          _name('R', 'V0D'),
                          weight=0.70)

        self.net.add_edge(_name('L', 'RG_F'),
                          _name('L', 'V3'),
                          weight=0.35)

        self.net.add_edge(_name('R', 'RG_F'),
                          _name('R', 'V3'),
                          weight=0.35)

        self.net.add_edge(_name('L', 'RG_F'),
                          _name('L', 'V2a_diag'),
                          weight=0.50)

        self.net.add_edge(_name('R', 'RG_F'),
                          _name('R', 'V2a_diag'),
                          weight=0.50)

        self.net.add_edge(_name('L', 'CINi1'),
                          _name('R', 'RG_F'),
                          weight=-0.03)

        self.net.add_edge(_name('R', 'CINi1'),
                          _name('L', 'RG_F'),
                          weight=-0.03)

        self.net.add_edge(_name('L', 'V0V'),
                          _name('R', 'IniV0V'),
                          weight=0.60)

        self.net.add_edge(_name('R', 'V0V'),
                          _name('L', 'IniV0V'),
                          weight=0.60)

        self.net.add_edge(_name('L', 'V0D'),
                          _name('R', 'RG_F'),
                          weight=-0.07)

        self.net.add_edge(_name('R', 'V0D'),
                          _name('L', 'RG_F'),
                          weight=-0.07)

        self.net.add_edge(_name('L', 'V3'),
                          _name('R', 'RG_F'),
                          weight=0.03)

        self.net.add_edge(_name('R', 'V3'),
                          _name('L', 'RG_F'),
                          weight=0.03)

        self.net.add_edge(_name('L', 'V2a'),
                          _name('L', 'V0V'),
                          weight=1.0)

        self.net.add_edge(_name('R', 'V2a'),
                          _name('R', 'V0V'),
                          weight=1.0)

        self.net.add_edge(_name('L', 'IniV0V'),
                          _name('L', 'RG_F'),
                          weight=-0.07)

        self.net.add_edge(_name('R', 'IniV0V'),
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
                          _name('L', 'Sh2_hom_fh'),
                          weight=0.50)

        self.net.add_edge(_name('R', 'RG_E', 'F'),
                          _name('R', 'Sh2_hom_fh'),
                          weight=0.50)

        self.net.add_edge(_name('L', 'RG_E', 'H'),
                          _name('L', 'Sh2_hom_hf'),
                          weight=0.50)

        self.net.add_edge(_name('R', 'RG_E', 'H'),
                          _name('R', 'Sh2_hom_hf'),
                          weight=0.50)

        self.net.add_edge(_name('L', 'RG_F', 'F'),
                          _name('L', 'Ini_hom_fh'),
                          weight=0.70)

        self.net.add_edge(_name('R', 'RG_F', 'F'),
                          _name('R', 'Ini_hom_fh'),
                          weight=0.70)

        self.net.add_edge(_name('L', 'RG_F', 'F'),
                          _name('L', 'V0D_diag'),
                          weight=0.50)

        self.net.add_edge(_name('R', 'RG_F', 'F'),
                          _name('R', 'V0D_diag'),
                          weight=0.50)

        self.net.add_edge(_name('L', 'V2a_diag', 'F'),
                          _name('L', 'V0V_diag_fh'),
                          weight=0.90)

        self.net.add_edge(_name('R', 'V2a_diag', 'F'),
                          _name('R', 'V0V_diag_fh'),
                          weight=0.90)

        self.net.add_edge(_name('L', 'V2a_diag', 'H'),
                          _name('L', 'V0V_diag_hf'),
                          weight=0.90)

        self.net.add_edge(_name('R', 'V2a_diag', 'H'),
                          _name('R', 'V0V_diag_hf'),
                          weight=0.90)

        self.net.add_edge(_name('L', 'V0D_diag'),
                          _name('R', 'RG_F', 'H'),
                          weight=-0.075)

        self.net.add_edge(_name('R', 'V0D_diag'),
                          _name('L', 'RG_F', 'H'),
                          weight=-0.075)

        self.net.add_edge(_name('L', 'V0V_diag_fh'),
                          _name('R', 'RG_F', 'H'),
                          weight=0.02)

        self.net.add_edge(_name('R', 'V0V_diag_fh'),
                          _name('L', 'RG_F', 'H'),
                          weight=0.02)

        self.net.add_edge(_name('L', 'V0V_diag_hf'),
                          _name('R', 'RG_F', 'F'),
                          weight=0.065)

        self.net.add_edge(_name('R', 'V0V_diag_hf'),
                          _name('L', 'RG_F', 'F'),
                          weight=0.065)

        self.net.add_edge(_name('L', 'Ini_hom_fh'),
                          _name('L', 'RG_F', 'H'),
                          weight=-0.01)

        self.net.add_edge(_name('R', 'Ini_hom_fh'),
                          _name('R', 'RG_F', 'H'),
                          weight=-0.01)

        self.net.add_edge(_name('L', 'Sh2_hom_fh'),
                          _name('L', 'RG_F', 'H'),
                          weight=0.01)

        self.net.add_edge(_name('R', 'Sh2_hom_fh'),
                          _name('R', 'RG_F', 'H'),
                          weight=0.01)

        self.net.add_edge(_name('L', 'Sh2_hom_hf'),
                          _name('L', 'RG_F', 'F'),
                          weight=0.125)

        self.net.add_edge(_name('R', 'Sh2_hom_hf'),
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
