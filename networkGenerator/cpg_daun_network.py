""" Danner CPG Model. """

import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import biolog
from daun_net_gen import CPG
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
    net1 = CPG('PR_L1', anchor_x=0., anchor_y=0.)  #: Directed graph

#: Connecting sub graphs
    net = nx.compose_all([net1.cpg])

    #: Connect Nodes Between Sub-Networks

    nx.write_graphml(net,
                     './conf/auto_gen_daun_cpg.graphml')

    #: Initialize network
    net_ = NetworkGenerator('./conf/auto_gen_daun_cpg.graphml')

    #: initialize network parameters
    #: pylint: disable=invalid-name
    dt = 1  #: Time step
    time_vec = np.arange(0, 10000, dt)  #: Time
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
    plt.subplot(211)
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
