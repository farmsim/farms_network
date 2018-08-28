""" Danner CPG Model. eLife """

import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import rc

import biolog
from danner_net_gen_e_life import (CPG, LPSN, Commissural, ConnectFore2Hind,
                                   ConnectRG2Commissural)
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
    net1 = CPG('FL', anchor_x=-10., anchor_y=-10.)  #: Directed graph
    net2 = CPG('FR', anchor_x=10., anchor_y=-10.)  #: Directed graph
    net3 = CPG('HL', anchor_x=-10., anchor_y=15.)  #: Directed graph
    net4 = CPG('HR', anchor_x=10, anchor_y=15.)  #: Directed graph

    #: Commussiral
    net5 = Commissural('FL', anchor_x=-3, anchor_y=-10.,
                       color='c')  #: Directed graph
    net6 = Commissural('FR', anchor_x=3, anchor_y=-10.,
                       color='c')  #: Directed graph
    net7 = Commissural('HL', anchor_x=-3, anchor_y=15.,
                       color='c')  #: Directed graph
    net8 = Commissural('HR', anchor_x=3, anchor_y=15.,
                       color='c')  #: Directed graph

    #: Ipsilateral
    net9 = LPSN('L', anchor_x=-3., anchor_y=4.,
                color='c')  #: Directed graph
    net10 = LPSN('R', anchor_x=3., anchor_y=4.,
                 color='c')  #: Directed graph

    #: Connecting sub graphs

    net_RG_CIN1 = ConnectRG2Commissural(rg_l=net1.cpg, rg_r=net2.cpg,
                                        comm_l=net5.commissural,
                                        comm_r=net6.commissural)
    net_RG_CIN2 = ConnectRG2Commissural(rg_l=net3.cpg, rg_r=net4.cpg,
                                        comm_l=net7.commissural,
                                        comm_r=net8.commissural)

    net = ConnectFore2Hind(net_RG_CIN1.net,
                           net_RG_CIN2.net, net9.lpsn,
                           net10.lpsn)

    net = net.net

    nx.write_graphml(net,
                     './conf/auto_gen_danner_cpg.graphml')

    #: Initialize network
    net_ = NetworkGenerator('./conf/auto_gen_danner_cpg.graphml')

    #: initialize network parameters
    #: pylint: disable=invalid-name
    dt = 1  #: Time step
    time_vec = np.arange(0, 60, dt)  #: Time
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
    net_.visualize_network(plt)  #: Visualize network using Matplotlib

    _, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex='all')
    ax1.plot(time_vec*0.001,
             res[:, net_.dae.x.get_idx('V_FR_RG_F')], 'b')
    ax1.grid('on', axis='x')
    ax1.set_ylabel('FR')
    ax2.plot(time_vec*0.001,
             res[:, net_.dae.x.get_idx('V_FL_RG_F')], 'g')
    ax2.grid('on', axis='x')
    ax2.set_ylabel('FL')
    ax3.plot(time_vec*0.001, res[:, net_.dae.x.get_idx('V_HR_RG_F')],
             'r')
    ax3.grid('on', axis='x')
    ax3.set_ylabel('HR')
    ax4.plot(time_vec*0.001, res[:, net_.dae.x.get_idx('V_HL_RG_F')],
             'k')
    ax4.grid('on', axis='x')
    ax4.set_ylabel('HL')
    ax5.fill_between(time_vec*0.001, 0, alpha,
                     color=(0.2, 0.2, 0.2), alpha=0.5)
    ax5.grid('on', axis='x')
    ax5.set_ylabel('ALPHA')
    ax5.set_xlabel('Time [s]')

    plt.show()


if __name__ == '__main__':
    main()
