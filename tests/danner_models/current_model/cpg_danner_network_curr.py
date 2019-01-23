""" Danner CPG Model. Current Model """

import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import rc

import biolog
from danner_net_gen_curr import (CPG, PatternFormation, Commissural,
                                 HomoLateral, ContraLateral,
                                 Motorneurons,
                                 ConnectPF2RG,
                                 ConnectRG2Commissural,
                                 ConnectFore2Hind)

from network_generator.network_generator import NetworkGenerator

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels


def main():
    """Main."""

    #: CPG
    net_cpg1 = CPG('HL', anchor_x=0., anchor_y=40.)  #: Directed graph
    net_cpg2 = CPG('HR', anchor_x=40., anchor_y=40.)  #: Directed graph
    net_cpg3 = CPG('FL', anchor_x=0., anchor_y=-20.)  #: Directed graph
    net_cpg4 = CPG('FR', anchor_x=40., anchor_y=-20.)  #: Directed graph

    #: PATTERN FORMATION LAYER
    net_pf1 = PatternFormation('HL', anchor_x=0.,
                               anchor_y=45.)  #: Directed graph
    net_pf2 = PatternFormation(
        'HR', anchor_x=40., anchor_y=45.)  #: Directed graph
    net_pf3 = PatternFormation('FL', anchor_x=0.,
                               anchor_y=-15.)  #: Directed graph
    net_pf4 = PatternFormation(
        'FR', anchor_x=40., anchor_y=-15.)  #: Directed graph

    #: Commissural
    net_comm1 = Commissural('HL', anchor_x=10., anchor_y=20.)
    net_comm2 = Commissural('HR', anchor_x=30., anchor_y=20.)
    net_comm3 = Commissural('FL', anchor_x=10., anchor_y=10.)
    net_comm4 = Commissural('FR', anchor_x=30., anchor_y=10.)

    #: HomoLateral
    net_homo_l = HomoLateral('L', anchor_x=17.2, anchor_y=20.)
    net_homo_r = HomoLateral('R', anchor_x=22.5, anchor_y=20.)
    net_homo = nx.compose_all([net_homo_l.homo_net,
                               net_homo_r.homo_net])

    #: ContraLateral
    net_contra1 = ContraLateral('HL', anchor_x=15., anchor_y=20.)
    net_contra2 = ContraLateral('HR', anchor_x=25., anchor_y=20.)
    net_contra3 = ContraLateral('FL', anchor_x=15., anchor_y=10.)
    net_contra4 = ContraLateral('FR', anchor_x=25., anchor_y=10.)
    net_contra = nx.compose_all([net_contra1.contra_net,
                                 net_contra2.contra_net,
                                 net_contra3.contra_net,
                                 net_contra4.contra_net])

    #: Motorneurons
    hind_muscles = ['PMA', 'CF', 'SM', 'POP', 'RF', 'TA', 'SOL', 'LG']
    net_motorneurons_hl = Motorneurons('HL', hind_muscles, anchor_x=0.,
                                       anchor_y=60.)
    net_motorneurons_hr = Motorneurons('HR', hind_muscles, anchor_x=40.,
                                       anchor_y=60.)

    #: Network Connections
    net_rg1_pf1 = ConnectPF2RG(net_cpg1.cpg_net, net_pf1.pf_net)
    net_rg2_pf2 = ConnectPF2RG(net_cpg2.cpg_net, net_pf2.pf_net)
    net_rg3_pf3 = ConnectPF2RG(net_cpg3.cpg_net, net_pf3.pf_net)
    net_rg4_pf4 = ConnectPF2RG(net_cpg4.cpg_net, net_pf4.pf_net)

    net_rg1_comm1 = ConnectRG2Commissural(net_rg1_pf1.net,
                                          net_rg2_pf2.net,
                                          net_comm1.comm_net,
                                          net_comm2.comm_net)

    net_rg2_comm2 = ConnectRG2Commissural(net_rg3_pf3.net,
                                          net_rg4_pf4.net,
                                          net_comm3.comm_net,
                                          net_comm4.comm_net)

    net_pf1_comm1 = ConnectRG2Commissural(net_rg1_pf1.net,
                                          net_rg2_pf2.net,
                                          net_comm1.comm_net,
                                          net_comm2.comm_net)

    net_pf2_comm2 = ConnectRG2Commissural(net_rg3_pf3.net,
                                          net_rg4_pf4.net,
                                          net_comm3.comm_net,
                                          net_comm4.comm_net)

    net_fore = nx.compose_all([net_rg1_comm1.net, net_pf1_comm1.net])
    net_hind = nx.compose_all([net_rg2_comm2.net, net_pf2_comm2.net])

    net = ConnectFore2Hind(net_fore, net_hind, net_homo,
                           net_contra)

    net = nx.compose_all([net.net, net_motorneurons_hl.mn_net,
                          net_motorneurons_hr.mn_net])

    #: Location to save the network
    net_dir = os.path.join(
        os.path.dirname(__file__),
        './conf/auto_gen_danner_cpg.graphml')
    try:
        nx.write_graphml(net, net_dir)
    except IOError:
        if not os.path.isdir(os.path.split(net_dir)[0]):
            biolog.info('Creating directory : {}'.format(net_dir))
            os.mkdir(os.path.split(net_dir)[0])
            nx.write_graphml(net, net_dir)
        else:
            biolog.error('Error in creating directory!')
            raise IOError()

    #: Initialize network
    net_ = NetworkGenerator('./conf/auto_gen_danner_cpg.graphml')

    #: initialize network parameters
    #: pylint: disable=invalid-name
    dt = 1  #: Time step
    time_vec = np.arange(0, 1, dt)  #: Time

    #: Vector to store results
    res = np.empty([len(time_vec), len(net_.dae.x)])

    #: opts
    opts = {'tf': dt,
            'jit': False,
            "enable_jacobian": True,
            "print_time": False,
            "print_stats": False,
            "reltol": 1e-4,
            "abstol": 1e-3}

    # #: Setup the integrator
    net_.setup_integrator(integration_method='cvodes',
                          opts=opts)

    #: Integrate the network
    biolog.info('Begin Integration!')

    #: Network drive : Alpha
    alpha = np.linspace(0, 1, len(time_vec))

    start_time = time.time()
    for idx, _ in enumerate(time_vec):
        net_.dae.u.set_all_val(alpha[idx])
        res[idx] = net_.step()['xf'].full()[:, 0]
    end_time = time.time()

    biolog.info('Execution Time : {}'.format(
        end_time - start_time))

    # #: Results
    net_.save_network_to_dot()
    net_.visualize_network(node_size=250,
                           node_labels=False,
                           edge_labels=False,
                           edge_alpha=False,
                           plt_out=plt)  #: Visualize network using Matplotlib

    # _, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex='all')
    # ax1.plot(time_vec*0.001,
    #          res[:, net_.dae.x.get_idx('V_FL_RG_F')], 'b')
    # ax1.grid('on', axis='x')
    # ax1.set_ylabel('FL')
    # # ax2.plot(time_vec*0.001,
    # #          res[:, net_.dae.x.get_idx('V_FL_RG_F')], 'g')
    # # ax2.grid('on', axis='x')
    # # ax2.set_ylabel('FL')
    # # ax3.plot(time_vec*0.001, res[:, net_.dae.x.get_idx('V_HR_RG_F')],
    # #          'r')
    # # ax3.grid('on', axis='x')
    # # ax3.set_ylabel('HR')
    # # ax4.plot(time_vec*0.001, res[:, net_.dae.x.get_idx('V_HL_RG_F')],
    # #          'k')
    # # ax4.grid('on', axis='x')
    # # ax4.set_ylabel('HL')
    # ax5.fill_between(time_vec*0.001, 0, alpha,
    #                  color=(0.2, 0.2, 0.2), alpha=0.5)
    # ax5.grid('on', axis='x')
    # ax5.set_ylabel('ALPHA')
    # ax5.set_xlabel('Time [s]')

    plt.show()


if __name__ == '__main__':
    main()
