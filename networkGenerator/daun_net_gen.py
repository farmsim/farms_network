""" Generate Danner Network."""

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
        self.cpg.add_node(self.name+'_C1',
                          model='lif_daun_interneuron',
                          x=-1.0+anchor_x,
                          y=0.0+anchor_y,
                          color='b',
                          g_app=0.2,
                          e_app=0.0,
                          eps=0.0023,
                          c_m=0.9153,
                          v0=-70.0,
                          h0=0.8)
        self.cpg.add_node(self.name+'_C2',
                          model='lif_daun_interneuron',
                          x=1.0+anchor_x,
                          y=0.0+anchor_y,
                          color='b',
                          g_app=0.2,
                          e_app=0.0,
                          eps=0.0023,
                          c_m=0.9153,
                          v0=-10.0,
                          h0=0.1)

    def add_connections(self):
        self.cpg.add_edge(self.name+'_C1',
                          self.name+'_C2',
                          weight=-1.0,
                          g_syn=1.0,
                          e_syn=-80.0,
                          v_h_s=-43.0,
                          gamma_s=-10.0)
        self.cpg.add_edge(self.name+'_C2',
                          self.name+'_C1',
                          weight=-1.0,
                          g_syn=1.0,
                          e_syn=-80.0,
                          v_h_s=-43.0,
                          gamma_s=-10.0)
        return


class Interneurons(object):
    """Interneuons Network template.
    """

    def __init__(self, name, anchor_x=0.0, anchor_y=0.0, color='y'):
        """ Initialization. """
        super(Interneurons, self).__init__()
        self.interneurons = nx.DiGraph()
        self.name = name
        self.interneurons.name = name

        #: Methods
        self.add_neurons(anchor_x, anchor_y, color)
        self.add_connections()
        return

    def add_neurons(self, anchor_x, anchor_y, color):
        """ Add neurons. """
        self.interneurons.add_node(self.name+'_IN1',
                                   model='lif_daun_interneuron',
                                   x=-3.0+anchor_x,
                                   y=-2.0+anchor_y,
                                   color=color,
                                   eps=0.01,
                                   c_m=0.21,
                                   g_app=1.6,
                                   e_app=-80.0,
                                   v0=-63.46,
                                   h0=0.7910)
        self.interneurons.add_node(self.name+'_IN2',
                                   model='lif_daun_interneuron',
                                   x=-5.0+anchor_x,
                                   y=-2.0+anchor_y,
                                   color=color,
                                   eps=0.01,
                                   c_m=0.21,
                                   g_app=1.6,
                                   e_app=-80.0,
                                   v0=-63.46,
                                   h0=0.7910)
        self.interneurons.add_node(self.name+'_IN3',
                                   model='lif_daun_interneuron',
                                   x=3.0+anchor_x,
                                   y=-2.0+anchor_y,
                                   color=color,
                                   eps=0.01,
                                   c_m=0.21,
                                   g_app=1.6,
                                   e_app=-80.0,
                                   v0=-63.46,
                                   h0=0.7910)
        self.interneurons.add_node(self.name+'_IN4',
                                   model='lif_daun_interneuron',
                                   x=5.0+anchor_x,
                                   y=-2.0+anchor_y,
                                   color=color,
                                   eps=0.01,
                                   c_m=0.21,
                                   g_app=1.6,
                                   e_app=-80.0,
                                   v0=-63.46,
                                   h0=0.7910)
        self.interneurons.add_node(self.name+'_IN5',
                                   model='lif_daun_interneuron',
                                   x=1.0+anchor_x,
                                   y=3.0+anchor_y,
                                   color=color,
                                   eps=0.01,
                                   c_m=0.21,
                                   g_app=1.6,
                                   e_app=-80.0,
                                   v0=-63.46,
                                   h0=0.7910)
        self.interneurons.add_node(self.name+'_IN6',
                                   model='lif_daun_interneuron',
                                   x=-1.0+anchor_x,
                                   y=3.0+anchor_y,
                                   color=color,
                                   eps=0.01,
                                   c_m=0.21,
                                   g_app=1.6,
                                   e_app=-80.0,
                                   v0=-63.46,
                                   h0=0.7910)

        return

    def add_connections(self):
        self.interneurons.add_edge(self.name+'_IN6',
                                   self.name+'_IN5',
                                   weight=1.0,
                                   g_syn=0.4,
                                   e_syn=0.0,
                                   v_h_s=-43.0,
                                   gamma_s=-10.0)
        return


class Motorneurons(object):
    """Interneuons Network template.
    """

    def __init__(self, name, anchor_x=0.0, anchor_y=0.0, color='c'):
        """ Initialization. """
        super(Motorneurons, self).__init__()
        self.motorneurons = nx.DiGraph()
        self.name = name
        self.motorneurons.name = name

        #: Methods
        self.add_neurons(anchor_x, anchor_y, color)
        self.add_connections()
        return

    def add_neurons(self, anchor_x, anchor_y, color):
        """ Add neurons. """
        self.motorneurons.add_node(self.name+'_MN1',
                                   model='lif_daun_motorneruon',
                                   x=-3.0+anchor_x,
                                   y=1.5+anchor_y,
                                   color=color,
                                   g_app=0.19,
                                   e_app=0.0,
                                   v0=-65.0,
                                   m_na0=0.9,
                                   h_na0=0.0,
                                   mm_k0=0.0,
                                   m_q0=0.0)

        self.motorneurons.add_node(self.name+'_MN2',
                                   model='lif_daun_motorneruon',
                                   x=-5.0+anchor_x,
                                   y=1.5+anchor_y,
                                   color=color,
                                   g_app=1.6,
                                   e_app=-80.0,
                                   v0=-65.0,
                                   m_na0=0.9,
                                   h_na0=0.0,
                                   mm_k0=0.0,
                                   m_q0=0.0)

        self.motorneurons.add_node(self.name+'_MN3',
                                   model='lif_daun_motorneruon',
                                   x=3.0+anchor_x,
                                   y=1.5+anchor_y,
                                   color=color,
                                   g_app=0.19,
                                   e_app=0.0,
                                   v0=-65.0,
                                   m_na0=0.9,
                                   h_na0=0.0,
                                   mm_k0=0.0,
                                   m_q0=0.0)

        self.motorneurons.add_node(self.name+'_MN4',
                                   model='lif_daun_motorneruon',
                                   x=5.0+anchor_x,
                                   y=1.5+anchor_y,
                                   color=color,
                                   g_app=0.19,
                                   e_app=0.0,
                                   v0=-65.0,
                                   m_na0=0.9,
                                   h_na0=0.0,
                                   mm_k0=0.0,
                                   m_q0=0.0)
        return

    def add_connections(self):
        pass


class ConnectCPG2Interneurons(object):
    """Connect a CPG circuit with Interneuons

    """

    def __init__(self, cpg, interneurons):
        """ Initialization. """
        super(ConnectCPG2Interneurons, self).__init__()
        self.net = nx.compose_all([cpg,
                                   interneurons])
        self.name = self.net.name

        #: Methods
        self.connect_circuits()
        return

    def connect_circuits(self):
        """ Connect CPG's to Interneurons. """

        def _name(name):
            """ Add the network name to the neuron."""
            return self.name + '_' + name

        self.net.add_edge(_name('C1'), _name('IN1'),
                          weight=1.0,
                          g_syn=0.5,
                          e_syn=0.0,
                          v_h_s=-43.0,
                          gamma_s=-10.0)

        self.net.add_edge(_name('C1'), _name('IN2'),
                          weight=1.0,
                          g_syn=0.5,
                          e_syn=0.0,
                          v_h_s=-43.0,
                          gamma_s=-10.0)

        self.net.add_edge(_name('C2'), _name('IN3'),
                          weight=1.0,
                          g_syn=0.5,
                          e_syn=0.0,
                          v_h_s=-43.0,
                          gamma_s=-10.0)

        self.net.add_edge(_name('C2'), _name('IN4'),
                          weight=1.0,
                          g_syn=0.5,
                          e_syn=0.0,
                          v_h_s=-43.0,
                          gamma_s=-10.0)

        self.net.add_edge(_name('IN5'), _name('C2'),
                          weight=-1.0,
                          g_syn=0.5,
                          e_syn=-80.0,
                          v_h_s=-43.0,
                          gamma_s=-10.0)

        self.net.add_edge(_name('IN6'), _name('C1'),
                          weight=1.0,
                          g_syn=0.1,
                          e_syn=0.0,
                          v_h_s=-43.0,
                          gamma_s=-10.0)
        return self.net


class ConnectInterneurons2Motorneurons(object):
    """Connect a Interneurons circuit with Motorneurons

    """

    def __init__(self, net, motorneurons):
        """ Initialization. """
        super(ConnectInterneurons2Motorneurons, self).__init__()
        self.net = nx.compose_all([net,
                                   motorneurons])
        self.name = self.net.name

        #: Methods
        self.connect_circuits()
        return

    def connect_circuits(self):
        """ Connect CPG's to Interneurons. """

        def _name(name):
            """ Add the network name to the neuron."""
            return self.name + '_' + name

        self.net.add_edge(_name('IN1'), _name('MN1'),
                          weight=-1.0,
                          g_syn=0.25,
                          e_syn=-80.0,
                          v_h_s=-43.0,
                          gamma_s=-0.1)

        self.net.add_edge(_name('IN2'), _name('MN2'),
                          weight=-1.0,
                          g_syn=0.25,
                          e_syn=-80.0,
                          v_h_s=-43.0,
                          gamma_s=-0.1)

        self.net.add_edge(_name('IN3'), _name('MN3'),
                          weight=-1.0,
                          g_syn=0.25,
                          e_syn=-80.0,
                          v_h_s=-43.0,
                          gamma_s=-0.1)

        self.net.add_edge(_name('IN4'), _name('MN4'),
                          weight=-1.0,
                          g_syn=0.25,
                          e_syn=-80.0,
                          v_h_s=-43.0,
                          gamma_s=-0.1)
        return self.net


def main():
    """ Main. """

    net = CPG('FORE')  #: Directed graph
    nx.write_graphml(net.cpg, './conf/auto_gen_danner_cpg.graphml')

    return


if __name__ == '__main__':
    main()
