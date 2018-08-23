""" Danner CPG Model. """

import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import biolog
from daun_net_gen import (CPG, ConnectCPG2Interneurons, Interneurons,
                          Motorneurons)
from network_generator import NetworkGenerator

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
    net1 = CPG('PR_L1', anchor_x=0., anchor_y=12.)
    net2 = CPG('PR_L2', anchor_x=0., anchor_y=0.)
    net3 = CPG('PR_L3', anchor_x=0., anchor_y=-12.)

    net4 = CPG('LD_L1', anchor_x=12, anchor_y=12.)
    net5 = CPG('LD_L2', anchor_x=12, anchor_y=0.)
    net6 = CPG('LD_L3', anchor_x=12, anchor_y=-12.)

    net7 = CPG('EF_L1', anchor_x=24, anchor_y=12.)
    net8 = CPG('EF_L2', anchor_x=24, anchor_y=0.)
    net9 = CPG('EF_L3', anchor_x=24, anchor_y=-12.)

    #: Interneurons
    net10 = Interneurons('PR_L1', anchor_x=0., anchor_y=12.)
    net11 = Interneurons('PR_L2', anchor_x=0., anchor_y=0.)
    net12 = Interneurons('PR_L3', anchor_x=0., anchor_y=-12.)

    net13 = Interneurons('LD_L1', anchor_x=12, anchor_y=12.)
    net14 = Interneurons('LD_L2', anchor_x=12, anchor_y=0.)
    net15 = Interneurons('LD_L3', anchor_x=12, anchor_y=-12.)

    net16 = Interneurons('EF_L1', anchor_x=24, anchor_y=12.)
    net17 = Interneurons('EF_L2', anchor_x=24, anchor_y=0.)
    net18 = Interneurons('EF_L3', anchor_x=24, anchor_y=-12.)

    #: Motorneurons
    net19 = Motorneurons('PR_L1', anchor_x=0., anchor_y=12.)
    net20 = Motorneurons('PR_L2', anchor_x=0., anchor_y=0.)
    net21 = Motorneurons('PR_L3', anchor_x=0., anchor_y=-12.)

    net22 = Motorneurons('LD_L1', anchor_x=12, anchor_y=12.)
    net23 = Motorneurons('LD_L2', anchor_x=12, anchor_y=0.)
    net24 = Motorneurons('LD_L3', anchor_x=12, anchor_y=-12.)

    net25 = Motorneurons('EF_L1', anchor_x=24, anchor_y=12.)
    net26 = Motorneurons('EF_L2', anchor_x=24, anchor_y=0.)
    net27 = Motorneurons('EF_L3', anchor_x=24, anchor_y=-12.)

    # Connect CPG to Interneurons
    net_C_IN_1 = ConnectCPG2Interneurons(net1.cpg, net10.interneurons)
    net_C_IN_2 = ConnectCPG2Interneurons(net2.cpg, net11.interneurons)
    net_C_IN_3 = ConnectCPG2Interneurons(net3.cpg, net12.interneurons)
    net_C_IN_4 = ConnectCPG2Interneurons(net4.cpg, net13.interneurons)
    net_C_IN_5 = ConnectCPG2Interneurons(net5.cpg, net14.interneurons)
    net_C_IN_6 = ConnectCPG2Interneurons(net6.cpg, net15.interneurons)
    net_C_IN_7 = ConnectCPG2Interneurons(net7.cpg, net16.interneurons)
    net_C_IN_8 = ConnectCPG2Interneurons(net8.cpg, net17.interneurons)
    net_C_IN_9 = ConnectCPG2Interneurons(net9.cpg, net18.interneurons)

#: Connecting sub graphs
    net = nx.compose_all([net_C_IN_1.net,
                          net_C_IN_2.net,
                          net_C_IN_3.net,
                          net_C_IN_4.net,
                          net_C_IN_5.net,
                          net_C_IN_6.net,
                          net_C_IN_7.net,
                          net_C_IN_8.net,
                          net_C_IN_9.net,
                          net19.motorneruons,
                          net21.motorneruons,
                          net22.motorneruons,
                          net23.motorneruons,
                          net24.motorneruons,
                          net25.motorneruons,
                          net26.motorneruons,
                          net27.motorneruons])

    #: Connect Nodes Between Sub-Networks

    nx.write_graphml(net,
                     './conf/auto_gen_daun_cpg.graphml')

    #: Initialize network
    net_ = NetworkGenerator('./conf/auto_gen_daun_cpg.graphml')

    #: initialize network parameters
    #: pylint: disable=invalid-name
    dt = 1  #: Time step
    time_vec = np.arange(0, 1000, dt)  #: Time
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
    net_.setup_integrator(opts=opts)

    #: Integrate the network
    biolog.info('Begin Integration!')

    biolog.info('PARAMETERS')
    print('\n'.join(
        ['{} : {}'.format(
            p.sym.name(), p.val) for p in net_.dae.p.param_list]))

    biolog.info('INPUTS')
    print('\n'.join(
        ['{} : {}'.format(
            p.sym.name(), p.val) for p in net_.dae.u.param_list]))

    biolog.info('INITIAL CONDITIONS')
    print('\n'.join(
        ['{} : {}'.format(
            p.sym.name(), p.val) for p in net_.dae.x.param_list]))

    biolog.info('CONSTANTS')
    print('\n'.join(
        ['{} : {}'.format(
            p.sym.name(), p.val) for p in net_.dae.c.param_list]))

    # start_time = time.time()
    # for idx, _ in enumerate(time_vec):
    #     res[idx] = net_.step()['xf'].full()[:, 0]
    # end_time = time.time()

    # biolog.info('Execution Time : {}'.format(
    #     end_time - start_time))

    # #: Results
    net_.save_network_to_dot()
    net_.visualize_network(plt)  #: Visualize network using Matplotlib

    plt.figure()
    plt.title('DAUNS NETWORK')
    plt.plot(time_vec*0.001,
             res[:, [net_.dae.x.get_idx('V_PR_L1_C1')]])
    plt.plot(time_vec*0.001,
             res[:, [net_.dae.x.get_idx('V_PR_L1_C2')]],
             ':', markersize=5.)
    plt.legend(('V_PR_L1_C1', 'V_PR_L1_C2'))
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
