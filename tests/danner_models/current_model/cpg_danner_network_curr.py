""" Danner CPG Model. Current Model """

import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import rc

import biolog
from danner_net_gen_curr import (CPG, LPSN, Commissural, ConnectFore2Hind,
                                 ConnectRG2Commissural, PatternFormation)
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
    net1 = CPG('HL', anchor_x=0., anchor_y=40.)  #: Directed graph
    net2 = CPG('HR', anchor_x=40., anchor_y=40.)  #: Directed graph
    net3 = CPG('FL', anchor_x=0., anchor_y=-20.)  #: Directed graph
    net4 = CPG('FR', anchor_x=40., anchor_y=-20.)  #: Directed graph

    #: PATTERN FORMATION LAYER
    net5 = PatternFormation('HL', anchor_x=0.,
                            anchor_y=45.)  #: Directed graph
    net6 = PatternFormation(
        'HR', anchor_x=40., anchor_y=45.)  #: Directed graph
    net7 = PatternFormation('FL', anchor_x=0.,
                            anchor_y=-15.)  #: Directed graph
    net8 = PatternFormation(
        'FR', anchor_x=40., anchor_y=-15.)  #: Directed graph

    #: Commissural
    net9 = Commissural('HL', anchor_x=10., anchor_y=40.)
    net10 = Commissural('HR', anchor_x=30., anchor_y=40.)
    net11 = Commissural('FL', anchor_x=10., anchor_y=10.)
    net12 = Commissural('FR', anchor_x=30., anchor_y=10.)

    #: LPSN
    net13 = LPSN('L', anchor_x=15, anchor_y=25.)
    net14 = LPSN('R', anchor_x=25., anchor_y=25.)

    #: Network Connections
    net15 = ConnectRG2Commissural(net1.cpg, net2.cpg,
                                  net9.commissural, net10.commissural)
    net16 = ConnectRG2Commissural(net3.cpg, net4.cpg,
                                  net11.commissural, net12.commissural)
    net17 = ConnectFore2Hind(net15.net, net16.net, net13.lpsn, net14.lpsn)

    net = nx.compose_all([net5.pf_net, net6.pf_net, net7.pf_net, net8.pf_net,
                          net17.net])

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
    time_vec = np.arange(0, 6, dt)  #: Time

    #: Vector to store results
    res = np.empty([len(time_vec), len(net_.dae.x)])

    #: opts
    opts = {'tf': dt,
            'jit': False,
            "enable_jacobian": True,
            "print_time": False,
            "print_stats": False,
            "reltol": 1e-6,
            "abstol": 1e-6}

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
    net_.visualize_network(node_size=1250,
                           edge_labels=False,
                           plt_out=plt)  #: Visualize network using Matplotlib

    _, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex='all')
    ax1.plot(time_vec*0.001,
             res[:, net_.dae.x.get_idx('V_FL_RG_F')], 'b')
    ax1.grid('on', axis='x')
    ax1.set_ylabel('FL')
    # ax2.plot(time_vec*0.001,
    #          res[:, net_.dae.x.get_idx('V_FL_RG_F')], 'g')
    # ax2.grid('on', axis='x')
    # ax2.set_ylabel('FL')
    # ax3.plot(time_vec*0.001, res[:, net_.dae.x.get_idx('V_HR_RG_F')],
    #          'r')
    # ax3.grid('on', axis='x')
    # ax3.set_ylabel('HR')
    # ax4.plot(time_vec*0.001, res[:, net_.dae.x.get_idx('V_HL_RG_F')],
    #          'k')
    # ax4.grid('on', axis='x')
    # ax4.set_ylabel('HL')
    ax5.fill_between(time_vec*0.001, 0, alpha,
                     color=(0.2, 0.2, 0.2), alpha=0.5)
    ax5.grid('on', axis='x')
    ax5.set_ylabel('ALPHA')
    ax5.set_xlabel('Time [s]')

    plt.show()


if __name__ == '__main__':
    main()
