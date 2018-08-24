""" Danner CPG Model. """

import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from daun_net_gen import SideNetwork
from network_generator.network_generator import NetworkGenerator

import biolog

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

    #: Left side network
    net_right = SideNetwork('R', 60.0, 0.0)
    net_left = SideNetwork('L', 0.0, 0.0)

    net = nx.compose_all([net_left.net,
                          net_right.net])

    #: Connect Nodes Between Sub-Networks
    nx.write_graphml(net,
                     './conf/auto_gen_daun_cpg.graphml')

    #: Initialize network
    net_ = NetworkGenerator('./conf/auto_gen_daun_cpg.graphml')

    #: initialize network parameters
    #: pylint: disable=invalid-name
    dt = 1  #: Time step
    time_vec = np.arange(0, 100, dt)  #: Time
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
    net_.setup_integrator(integration_method='cvodes', opts=opts)

    #: Integrate the network
    biolog.info('Begin Integration!')

    # biolog.info('PARAMETERS')
    # print('\n'.join(
    #     ['{} : {}'.format(
    #         p.sym.name(), p.val) for p in net_.dae.p.param_list]))

    # biolog.info('INPUTS')
    # print('\n'.join(
    #     ['{} : {}'.format(
    #         p.sym.name(), p.val) for p in net_.dae.u.param_list]))

    # biolog.info('INITIAL CONDITIONS')
    # print('\n'.join(
    #     ['{} : {}'.format(
    #         p.sym.name(), p.val) for p in net_.dae.x.param_list]))

    # biolog.info('CONSTANTS')
    # print('\n'.join(
    #     ['{} : {}'.format(
    #         p.sym.name(), p.val) for p in net_.dae.c.param_list]))

    start_time = time.time()
    for idx, _ in enumerate(time_vec):
        res[idx] = net_.step()['xf'].full()[:, 0]
    end_time = time.time()

    biolog.info('Execution Time : {}'.format(
        end_time - start_time))

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
    # plt.plot(time_vec*0.001,
    #          res[:, [net_.dae.x.get_idx('V_PR_L1_IN6')]],
    #          ':', markersize=5.)
    plt.legend(('V_PR_L1_C1', 'V_PR_L1_C2',
                # 'V_PR_L1_IN6'
                ))
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
