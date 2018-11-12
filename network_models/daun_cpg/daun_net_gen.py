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
                          h0=0.8,
                          output=True)
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
                          h0=0.1,
                          output=True)

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
                                   g_leak=2.8,
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
                                   g_leak=2.8,
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
                                   g_leak=2.8,
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
                                   g_leak=2.8,
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
                                   g_leak=6.8,
                                   g_app=0.0,
                                   e_app=0.0,
                                   v0=-63.46,
                                   h0=0.7910)
        self.interneurons.add_node(self.name+'_IN6',
                                   model='lif_daun_interneuron',
                                   x=-1.0+anchor_x,
                                   y=3.0+anchor_y,
                                   color=color,
                                   eps=0.01,
                                   c_m=0.21,
                                   g_leak=10.,
                                   g_app=0.0,
                                   e_app=0.0,
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

        self.net.add_edge(_name('IN6'), _name('C2'),
                          weight=-1.0,
                          g_syn=0.05,
                          e_syn=-80.0,
                          v_h_s=-43.0,
                          gamma_s=-10.0)

        self.net.add_edge(_name('IN6'), _name('C1'),
                          weight=1.0,
                          g_syn=0.1,
                          e_syn=0.0,
                          v_h_s=-43.0,
                          gamma_s=-0.42)
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


class SideNetwork(object):
    """Daun's Insect One Side Network Class.
    """

    def __init__(self, side, anchor_x=0.0, anchor_y=0.0):
        super(SideNetwork, self).__init__()
        self.side = side
        self.net = None

        #: Method
        self._generate_network(anchor_x, anchor_y)

    def _generate_network(self, x_pos, y_pos):
        """ Generate the network. """
        #: CPG
        net1 = CPG('PR_{}1'.format(
            self.side), anchor_x=0.+x_pos, anchor_y=12.+y_pos)
        net2 = CPG('PR_{}2'.format(
            self.side), anchor_x=0.+x_pos, anchor_y=0.+y_pos)
        net3 = CPG('PR_{}3'.format(
            self.side), anchor_x=0.+x_pos, anchor_y=-12.+y_pos)

        net4 = CPG('LD_{}1'.format(
            self.side), anchor_x=12+x_pos, anchor_y=12.+y_pos)
        net5 = CPG('LD_{}2'.format(
            self.side), anchor_x=12+x_pos, anchor_y=0.+y_pos)
        net6 = CPG('LD_{}3'.format(
            self.side), anchor_x=12+x_pos, anchor_y=-12.+y_pos)

        net7 = CPG('EF_{}1'.format(
            self.side), anchor_x=24+x_pos, anchor_y=12.+y_pos)
        net8 = CPG('EF_{}2'.format(
            self.side), anchor_x=24+x_pos, anchor_y=0.+y_pos)
        net9 = CPG('EF_{}3'.format(
            self.side), anchor_x=24+x_pos, anchor_y=-12.+y_pos)

        #: Interneurons
        net10 = Interneurons('PR_{}1'.format(
            self.side), anchor_x=0.+x_pos, anchor_y=12.+y_pos)
        net11 = Interneurons('PR_{}2'.format(
            self.side), anchor_x=0.+x_pos, anchor_y=0.+y_pos)
        net12 = Interneurons('PR_{}3'.format(
            self.side), anchor_x=0.+x_pos, anchor_y=-12.+y_pos)

        net13 = Interneurons('LD_{}1'.format(
            self.side), anchor_x=12+x_pos, anchor_y=12.+y_pos)
        net14 = Interneurons('LD_{}2'.format(
            self.side), anchor_x=12+x_pos, anchor_y=0.+y_pos)
        net15 = Interneurons('LD_{}3'.format(
            self.side), anchor_x=12+x_pos, anchor_y=-12.+y_pos)

        net16 = Interneurons('EF_{}1'.format(
            self.side), anchor_x=24+x_pos, anchor_y=12.+y_pos)
        net17 = Interneurons('EF_{}2'.format(
            self.side), anchor_x=24+x_pos, anchor_y=0.+y_pos)
        net18 = Interneurons('EF_{}3'.format(
            self.side), anchor_x=24+x_pos, anchor_y=-12.+y_pos)

        #: Motorneurons
        net19 = Motorneurons('PR_{}1'.format(
            self.side), anchor_x=0.+x_pos, anchor_y=12.+y_pos)
        net20 = Motorneurons('PR_{}2'.format(
            self.side), anchor_x=0.+x_pos, anchor_y=0.+y_pos)
        net21 = Motorneurons('PR_{}3'.format(
            self.side), anchor_x=0.+x_pos, anchor_y=-12.+y_pos)

        net22 = Motorneurons('LD_{}1'.format(
            self.side), anchor_x=12+x_pos, anchor_y=12.+y_pos)
        net23 = Motorneurons('LD_{}2'.format(
            self.side), anchor_x=12+x_pos, anchor_y=0.+y_pos)
        net24 = Motorneurons('LD_{}3'.format(
            self.side), anchor_x=12+x_pos, anchor_y=-12.+y_pos)

        net25 = Motorneurons('EF_{}1'.format(
            self.side), anchor_x=24+x_pos, anchor_y=12.+y_pos)
        net26 = Motorneurons('EF_{}2'.format(
            self.side), anchor_x=24+x_pos, anchor_y=0.+y_pos)
        net27 = Motorneurons('EF_{}3'.format(
            self.side), anchor_x=24+x_pos, anchor_y=-12.+y_pos)

        # Connect CPG to Interneurons
        #: pylint: disable=invalid-name
        net_C_IN_1 = ConnectCPG2Interneurons(net1.cpg, net10.interneurons)
        net_C_IN_2 = ConnectCPG2Interneurons(net2.cpg, net11.interneurons)
        net_C_IN_3 = ConnectCPG2Interneurons(net3.cpg, net12.interneurons)
        net_C_IN_4 = ConnectCPG2Interneurons(net4.cpg, net13.interneurons)
        net_C_IN_5 = ConnectCPG2Interneurons(net5.cpg, net14.interneurons)
        net_C_IN_6 = ConnectCPG2Interneurons(net6.cpg, net15.interneurons)
        net_C_IN_7 = ConnectCPG2Interneurons(net7.cpg, net16.interneurons)
        net_C_IN_8 = ConnectCPG2Interneurons(net8.cpg, net17.interneurons)
        net_C_IN_9 = ConnectCPG2Interneurons(net9.cpg, net18.interneurons)

        #: Connect Interneurons to Motorneurons
        net_IN_MN_1 = ConnectInterneurons2Motorneurons(
            net_C_IN_1.net, net19.motorneurons)
        net_IN_MN_2 = ConnectInterneurons2Motorneurons(
            net_C_IN_2.net, net20.motorneurons)
        net_IN_MN_3 = ConnectInterneurons2Motorneurons(
            net_C_IN_3.net, net21.motorneurons)
        net_IN_MN_4 = ConnectInterneurons2Motorneurons(
            net_C_IN_4.net, net22.motorneurons)
        net_IN_MN_5 = ConnectInterneurons2Motorneurons(
            net_C_IN_5.net, net23.motorneurons)
        net_IN_MN_6 = ConnectInterneurons2Motorneurons(
            net_C_IN_6.net, net24.motorneurons)
        net_IN_MN_7 = ConnectInterneurons2Motorneurons(
            net_C_IN_7.net, net25.motorneurons)
        net_IN_MN_8 = ConnectInterneurons2Motorneurons(
            net_C_IN_8.net, net26.motorneurons)
        net_IN_MN_9 = ConnectInterneurons2Motorneurons(
            net_C_IN_9.net, net27.motorneurons)

        #: Connecting sub graphs
        self.net = nx.compose_all([net_IN_MN_1.net,
                                   net_IN_MN_2.net,
                                   net_IN_MN_3.net,
                                   net_IN_MN_4.net,
                                   net_IN_MN_5.net,
                                   net_IN_MN_6.net,
                                   net_IN_MN_7.net,
                                   net_IN_MN_8.net,
                                   net_IN_MN_9.net
                                   ])
        return
