""" Generate Danner Network."""

import networkx as nx


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
                          color='b',
                          m_e=0.1,
                          b_e=0.0,
                          v0=-60.0,
                          h0=0.0)
        self.cpg.add_node(self.name+'_RG_E',
                          model='lif_danner_nap',
                          x=1.0+anchor_x,
                          y=4.0+anchor_y,
                          color='b',
                          m_e=0.0,
                          b_e=0.1,
                          v0=-60.0,
                          h0=0.0)
        self.cpg.add_node(self.name+'_In_F',
                          model='lif_danner',
                          x=0.0+anchor_x,
                          y=2.0+anchor_y,
                          color='y',
                          v0=-60.0)
        self.cpg.add_node(self.name+'_In_E',
                          model='lif_danner',
                          x=2.0+anchor_x,
                          y=2.0+anchor_y,
                          color='y',
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
        self.commissural.add_node(self.name+'_CINe1',
                                  model='lif_danner',
                                  x=1.0+anchor_x,
                                  y=-1.0+anchor_y,
                                  color=color,
                                  v0=-60.0)
        self.commissural.add_node(self.name+'_CINe2',
                                  model='lif_danner',
                                  x=1.0+anchor_x,
                                  y=1.0+anchor_y,
                                  color=color,
                                  v0=-60.0)
        self.commissural.add_node(self.name+'_CINi1',
                                  model='lif_danner',
                                  x=1.0+anchor_x,
                                  y=3.0+anchor_y,
                                  color=color,
                                  v0=-60.0)
        self.commissural.add_node(self.name+'_CINi2',
                                  model='lif_danner',
                                  x=1.0+anchor_x,
                                  y=5.0+anchor_y,
                                  color=color,
                                  v0=-60.0)
        self.commissural.add_node(self.name+'_Ini1',
                                  model='lif_danner',
                                  x=1.+anchor_x,
                                  y=2+anchor_y,
                                  color=color,
                                  v0=-60.0)
        return

    def add_connections(self):
        return


class Ipsilateral(object):
    """Ipsilateral Network template.

    """

    def __init__(self, name, anchor_x=0.0, anchor_y=0.0, color='c'):
        """ Initialization. """
        super(Ipsilateral, self).__init__()
        self.ipsilateral = nx.DiGraph()
        self.name = name
        self.ipsilateral.name = name

        #: Methods
        self.add_neurons(anchor_x, anchor_y, color)
        self.add_connections()
        return

    def add_neurons(self, anchor_x, anchor_y, color):
        """ Add neurons. """
        self.ipsilateral.add_node(self.name+'F_Ini_fh',
                                  model='lif_danner',
                                  x=0.0+anchor_x,
                                  y=0.0+anchor_y,
                                  color=color,
                                  v0=-60.0)

        self.ipsilateral.add_node(self.name+'H_Ini_fh',
                                  model='lif_danner',
                                  x=0.0+anchor_x,
                                  y=1.5+anchor_y,
                                  color=color,
                                  v0=-60.0)

        return

    def add_connections(self):
        return


def main():
    """ Main. """

    net = CPG('FORE')  #: Directed graph
    nx.write_graphml(net.cpg, './conf/auto_gen_danner_cpg.graphml')

    return


if __name__ == '__main__':
    main()
